import time
from pyspark.ml.clustering import GaussianMixture
from pyspark.sql import SparkSession

if __name__ == "__main__":

    start_computing_time = time.time()

    sparkConf = SparkSession \
        .builder \
        .appName("Gaussian Mixture Model") \
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

    # Train a Latent Dirichlet allocation.
    gmm = GaussianMixture(k=200, tol=0.0001,maxIter=10, seed=1)

    model=gmm.fit(trainingData)

    if model.hasSummary:
        summary=model.summary
        print("k=",summary.k)
        print("cluster sizes=",summary.clusterSizes)
        print("logLikelihood=",summary.logLikelihood)
        print("len weights=",len(model.weights))


    # Make predictions.
    predictions = model.transform(testData)
    predictions.show(5, truncate=False)

    #print("Gaussians shown as a DataFrame: ")
    #print(model.gaussiansDF.select("mean").head())
    #print(model.gaussiansDF.select("cov").head())

    total_computing_time = time.time() - start_computing_time
    print("Computing time: ", str(total_computing_time))