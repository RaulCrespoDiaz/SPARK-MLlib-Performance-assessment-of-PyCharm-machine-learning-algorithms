import time
from pyspark.ml.clustering import LDA
from pyspark.sql import SparkSession

if __name__ == "__main__":

    start_computing_time = time.time()

    sparkConf = SparkSession \
        .builder \
        .appName("LDA Model") \
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
    lda = LDA(k=2, maxIter=10)

    model=lda.fit(trainingData)

    ll = model.logLikelihood(data)
    lp = model.logPerplexity(data)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound on perplexity: " + str(lp))

    # Describe topics.
    topics = model.describeTopics(3)
    print("The topics described by their top-weighted terms:")
    topics.show(truncate=False)

    # Shows the result
    transformed = model.transform(data)
    transformed.show(truncate=False)

    total_computing_time = time.time() - start_computing_time
    print("Computing time: ", str(total_computing_time))