package models

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.log4j.Logger
import org.apache.log4j.Level

object RFModel {
  def main(args: Array[String]): Unit = {

    val t1 = System.nanoTime
    println("start time:", t1)

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val spark = SparkSession
      .builder()
      .appName("Random Forest Model")
      .config("spark.driver.memory", "8g")
      .master("local[1]")
      .getOrCreate()


    // Load the data stored in LIBSVM format as a DataFrame.
    val dataset = spark.read.format("libsvm").load("../data_libsvm.txt")

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(dataset)

    val dataindexed = labelIndexer.transform(dataset)

    val splits = dataindexed.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val numTraining = trainingData.count()
    val numTest = testData.count()
    println(s"numTraining = $numTraining, numTest = $numTest.")

    // Train a RandomForest model.
    //val numClasses = 2
    val numTrees = 3
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val rf = new RandomForestClassifier()
      .setMaxDepth(maxDepth)
      .setMaxBins(maxBins)
      .setImpurity(impurity)
      .setFeatureSubsetStrategy(featureSubsetStrategy)
      .setNumTrees(numTrees)
      .setLabelCol("indexedLabel")

    println("training data...")
    val model = rf.fit(trainingData)

    // Make predictions.
    println("predicting data...")
    val predictions = model.transform(testData)
    // Select example rows to display.
    predictions.show(5)

    // Select (prediction, true label) and compute test error.
    println("evaluating data...")
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val duration = (System.nanoTime - t1) / 1e9d
    println("Computing time: " + duration)
  }
}
