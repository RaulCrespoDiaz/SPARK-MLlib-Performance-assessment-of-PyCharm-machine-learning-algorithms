import time
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer

if __name__ == "__main__":

    start_computing_time = time.time()

    sparkConf = SparkSession \
        .builder \
        .appName("Decision Tree Model") \
        .config("spark.driver.memory", "8g") \
        .master("local[4]") \
        .getOrCreate()

    # Load the data stored in LIBSVM format as a DataFrame.
    data = sparkConf \
        .read \
        .format("libsvm") \
        .load("data\data_libsvm.txt")

    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    data = labelIndexer.transform(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    numTraining = trainingData.count()
    numTest = testData.count()
    print("numTraining = ",numTraining, " numTest =", numTest)

    # Train a RandomForest model.
    numTrees = 3
    featureSubsetStrategy = "auto"
    impurity = "gini"
    maxDepth = 4
    maxBins = 32

    dt = RandomForestClassifier() \
        .setMaxDepth(maxDepth) \
        .setMaxBins(maxBins) \
        .setImpurity(impurity) \
        .setFeatureSubsetStrategy(featureSubsetStrategy) \
        .setNumTrees(numTrees) \
        .setLabelCol("indexedLabel")

    model=dt.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)
    # Select example rows to display.
    predictions.show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))

    total_computing_time = time.time() - start_computing_time
    print("Computing time: ", str(total_computing_time))