package models

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

object KMModel {

  def main(args: Array[String]): Unit = {

    val t1 = System.nanoTime
    println("start time:", t1)

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val spark = SparkSession
      .builder()
      .appName("KMeans Clustering Model")
      .config("spark.driver.memory", "8g")
      .master("local[1]")
      .getOrCreate()


    // Load the data stored in LIBSVM format as a DataFrame.
    val dataset = spark.read.format("libsvm").load("../data_libsvm.txt")

    val splits = dataset.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Trains a k-means model.
    val kmeans = new KMeans().setK(200).setSeed(1L)
    val model = kmeans.fit(trainingData)

    // Make predictions
    val predictions = model.transform(testData)
    predictions.show(5)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)


    val duration = (System.nanoTime - t1) / 1e9d
    println("Computing time: " + duration)
  }

}
