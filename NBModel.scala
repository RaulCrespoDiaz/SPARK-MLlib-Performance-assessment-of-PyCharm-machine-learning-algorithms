package models

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

object NBModel {
  def main(args: Array[String]): Unit = {

    val t1 = System.nanoTime
    println("start time:", t1)

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val spark = SparkSession
      .builder()
      .appName("Naive Bayes Model")
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

    //val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")


    val modeltype = "multinomial"

    val nb = new NaiveBayes()
      .setModelType(modeltype)
      .setLabelCol("indexedLabel")

    println("training data...")
    val model = nb.fit(trainingData)

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
