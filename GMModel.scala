package models

import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

object GMModel {

  def main(args: Array[String]): Unit = {

    val t1 = System.nanoTime
    println("start time:", t1)

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val spark = SparkSession
      .builder()
      .appName("Gaussian Mixture Model")
      .config("spark.driver.memory", "8g")
      .master("local[8]")
      .getOrCreate()


    // Load the data stored in LIBSVM format as a DataFrame.
    val dataset = spark.read.format("libsvm").load("../data_libsvm.txt")


    val splits = dataset.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Trains Gaussian Mixture Model
    val gmm = new GaussianMixture().setK(200).setSeed(1).setMaxIter(10).setTol(0.0001)
    val model = gmm.fit(trainingData)

    if (model.hasSummary){
      val summary=model.summary
      println("k=",summary.k)
      println("cluster sizes=" + summary.clusterSizes)
      println("logLikelihood=" + summary.logLikelihood)
      println("len weights=" + model.weights.length)
    }
    // output parameters of mixture model
    //for (i <- 0 until model.getK) {
    //  println(s"Gaussian $i:\nweight=${model.weights(i)}\n" +
    //    s"mu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}\n")
    //}

    // Make predictions
    val predictions = model.transform(testData).select("features", "prediction")
    predictions.show(5,false)

    val duration = (System.nanoTime - t1) / 1e9d
    println("Computing time: " + duration)
  }

}
