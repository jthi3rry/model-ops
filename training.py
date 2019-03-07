import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import *

data_path = os.getenv('DATA_PATH', "hdfs:///tmp/data.csv")
model_path = os.getenv('MODEL_PATH', "hdfs:///tmp/model.spark")

# Create Spark Session
spark = SparkSession.builder.getOrCreate()

# Declare schemas
schema = StructType([
  StructField("BAD", DoubleType(), True),
  StructField("LOAN", DoubleType(), True),
  StructField("MORTDUE", DoubleType(), True),
  StructField("VALUE", DoubleType(), True),
  StructField("REASON", StringType(), True),
  StructField("JOB", StringType(), True),
  StructField("YOJ", DoubleType(), True),
  StructField("DEROG", DoubleType(), True),
  StructField("DELINQ", DoubleType(), True),
  StructField("CLAGE", DoubleType(), True),
  StructField("NINQ", DoubleType(), True),
  StructField("CLNO", DoubleType(), True),
  StructField("DEBTINC", DoubleType(), True),
])

data = spark.read.format("csv").option("header", "true").load(data_path, schema=schema).dropna()

(train, test) = data.randomSplit([0.7, 0.3])

numeric_cols = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG",
                "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]

categorical_cols = ["REASON", "JOB"]

label_indexer = StringIndexer(inputCol='BAD', outputCol='label')
reason_indexer = StringIndexer(inputCol='REASON', outputCol = 'REASON_indexed')
job_indexer = StringIndexer(inputCol='JOB', outputCol = 'JOB_indexed')

assembler = VectorAssembler(
    inputCols=['REASON_indexed', 'JOB_indexed'] + numeric_cols,
    outputCol='features')

classifier = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=10)
pipeline = Pipeline(stages=[reason_indexer, job_indexer, label_indexer, assembler, classifier])
model = pipeline.fit(train)

predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
print("The AUROC is %s " % (auroc))

model.write().overwrite().save(model_path)
