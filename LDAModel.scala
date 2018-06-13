package models

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

object LDAModel {

  def main(args: Array[String]): Unit = {

    val t1 = System.nanoTime
    println("start time:", t1)

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val spark = SparkSession
      .builder()
      .appName("Latent Dirichlet allocation Model")
			.master("local[8]")
      .getOrCreate()




    // Load the data stored in LIBSVM format as a DataFrame.
    val dataset = spark.read.format("libsvm").load("../data_libsvm.txt")

    val splits = dataset.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

	// Trains a LDA model.
	val lda = new LDA().setK(2).setMaxIter(10)
	val model = lda.fit(trainingData)

	val ll = model.logLikelihood(dataset)
	val lp = model.logPerplexity(dataset)
	println(s"The lower bound on the log likelihood of the entire corpus: $ll")
	println(s"The upper bound on perplexity: $lp")

	// Describe topics.
	val topics = model.describeTopics(3)
	println("The topics described by their top-weighted terms:")
	topics.show(false)

	// Shows the result.
	val prediction = model.transform(testData)
	prediction.show(false)


    val duration = (System.nanoTime - t1) / 1e9d
    println("Computing time: " + duration)
  }

}
