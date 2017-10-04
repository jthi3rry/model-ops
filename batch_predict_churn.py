#!/usr/bin/env python 

import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

model_path = os.getenv('MODEL_PATH', "hdfs:///tmp/churn.spark")
input_path = os.getenv('INPUT_PATH', "hdfs:///tmp/churn.all")
output_table= os.getenv('OUTPUT_TABLE', "thierry.churn_predictions")

# Create Spark Session
spark = SparkSession.builder.getOrCreate()

# Declare schemas and UDFs
from pyspark.sql.types import *
from pyspark.sql.functions import udf
prob_will_churn = udf(lambda v: float(v[1]), FloatType())

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
  # StructField("churned", StringType(), True)
])

# Restore model
model = PipelineModel.load(model_path)

# Load data from file
input_data = spark.read.csv(input_path, schema=schema)

# Predict
predictions = model.transform(input_data)

# Write predictions to table
predictions.select("state", "phone_number", "prediction", prob_will_churn('probability').alias("probability")) \
           .write.mode("overwrite").saveAsTable(output_table)

# Done