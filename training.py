import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import *

data_path = os.getenv('DATA_PATH', "hdfs:///tmp/churn.csv")
model_path = os.getenv('MODEL_PATH', "hdfs:///tmp/churn.spark")

# Create Spark Session
spark = SparkSession.builder.getOrCreate()

# Declare schemas
schema = StructType([
  StructField("state", StringType(), True),
  StructField("account_length", DoubleType(), True),
  StructField("area_code", StringType(), True),
  StructField("phone_number", StringType(), True),
  StructField("intl_plan", StringType(), True),
  StructField("voice_mail_plan", StringType(), True),
  StructField("number_vmail_messages", DoubleType(), True),
  StructField("total_day_minutes", DoubleType(), True),
  StructField("total_day_calls", DoubleType(), True),
  StructField("total_day_charge", DoubleType(), True),
  StructField("total_eve_minutes", DoubleType(), True),
  StructField("total_eve_calls", DoubleType(), True),
  StructField("total_eve_charge", DoubleType(), True),
  StructField("total_night_minutes", DoubleType(), True),
  StructField("total_night_calls", DoubleType(), True),
  StructField("total_night_charge", DoubleType(), True),
  StructField("total_intl_minutes", DoubleType(), True),
  StructField("total_intl_calls", DoubleType(), True),
  StructField("total_intl_charge", DoubleType(), True),
  StructField("number_customer_service_calls", DoubleType(), True),
  StructField("churned", StringType(), True)
])

data = spark.read.format("csv").option("header", "true").load(data_path, schema=schema).dropna()

(train, test) = data.randomSplit([0.7, 0.3])

numeric_cols = ["account_length", "number_vmail_messages", "total_day_minutes",
                "total_day_calls", "total_day_charge", "total_eve_minutes",
                "total_eve_calls", "total_eve_charge", "total_night_minutes",
                "total_night_calls", "total_night_charge", "total_intl_minutes",
                "total_intl_calls", "total_intl_charge","number_customer_service_calls"]

categorical_cols = ["state", "international_plan", "voice_mail_plan", "area_code"]

reduced_numeric_cols = ["account_length", "number_vmail_messages", "total_day_calls",
                        "total_day_charge", "total_eve_calls", "total_eve_charge",
                        "total_night_calls", "total_night_charge", "total_intl_calls",
                        "total_intl_charge","number_customer_service_calls"]

label_indexer = StringIndexer(inputCol='churned', outputCol='label')
plan_indexer = StringIndexer(inputCol='intl_plan', outputCol='intl_plan_indexed')

assembler = VectorAssembler(
    inputCols=['intl_plan_indexed'] + reduced_numeric_cols,
    outputCol='features')

classifier = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=10)
pipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, classifier])
model = pipeline.fit(train)

predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
print("The AUROC is %s " % (auroc))

model.write().overwrite().save(model_path)
