import time
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import ClusteringEvaluator

if __name__ == "__main__":

    start_computing_time = time.time()

    sparkConf = SparkSession \
        .builder \
        .appName("Kmeans Model") \
        .config("spark.driver.memory", "8g") \
        .master("local[8]") \
        .getOrCreate()

    # Load the data stored in LIBSVM format as a DataFrame.
    data = sparkConf \
        .read \
        .format("libsvm") \
        .load("data\data_libsvm.txt")
    data.show()

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    numTraining = trainingData.count()
    numTest = testData.count()
    print("numTraining = ",numTraining, " numTest =", numTest)

    # Train a KMeans model.
    dt = KMeans().setK(200).setSeed(1)

    model=dt.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)
    # Select example rows to display.
    predictions.show(5)

    # Select (prediction, true label) and compute test error
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    total_computing_time = time.time() - start_computing_time
    print("Computing time: ", str(total_computing_time))