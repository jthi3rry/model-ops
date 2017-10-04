#!/usr/bin/env python 
"""
Usage:

Stream producer:

  ncat -lk -p 49999

Example input:

  {"state":"WV","account_length":141.0,"area_code":" 415","phone_number":" 330-8173","intl_plan":" yes","voice_mail_plan":" yes","number_vmail_messages":37.0,"total_day_minutes":258.6,"total_day_calls":84.0,"total_day_charge":43.96,"total_eve_minutes":222.0,"total_eve_calls":111.0,"total_eve_charge":18.87,"total_night_minutes":326.4,"total_night_calls":97.0,"total_night_charge":14.69,"total_intl_minutes":11.2,"total_intl_calls":5.0,"total_intl_charge":3.02,"number_customer_service_calls":0.0}
  {"state":"IN","account_length":65.0,"area_code":" 415","phone_number":" 329-6603","intl_plan":" no","voice_mail_plan":" no","number_vmail_messages":0.0,"total_day_minutes":129.1,"total_day_calls":137.0,"total_day_charge":21.95,"total_eve_minutes":228.5,"total_eve_calls":83.0,"total_eve_charge":19.42,"total_night_minutes":208.8,"total_night_calls":111.0,"total_night_charge":9.4,"total_intl_minutes":12.7,"total_intl_calls":6.0,"total_intl_charge":3.43,"number_customer_service_calls":4.0}

Stream consumer:

  ./stream_predict_churn.py

Example output:

========= 2017-10-02 04:34:12 =========
========= 2017-10-02 04:34:13 =========
{"state":"WV","phone_number":" 330-8173","prediction":0.0,"probability":0.16069977}
{"state":"IN","phone_number":" 329-6603","prediction":1.0,"probability":0.71724945}
========= 2017-10-02 04:34:14 =========
========= 2017-10-02 04:34:15 =========
========= 2017-10-02 04:34:16 =========

"""

import os
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.streaming import StreamingContext

master = os.getenv('MASTER', "local[2]")
model_path = os.getenv('MODEL_PATH', "hdfs:///tmp/churn.spark")
input_host = os.getenv('INPUT_HOST', "localhost")
input_port = int(os.getenv('INPUT_PORT', "49999"))

# Create Spark Session
spark = SparkSession.builder.master(master).getOrCreate()

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

# Create streaming context
ssc = StreamingContext(spark.sparkContext, 1)

# Create DStream
dstream = ssc.socketTextStream(input_host, input_port)


def make_predictions(time, rdd):
  
  def decode_json_stream(l):
    return json.loads(l.decode('utf-8'))

  def writeln(l):
    print(l)
    
  print("========= %s =========" % str(time))

  # Convert rdd to DataFrame
  input_data = rdd.map(decode_json_stream).toDF(schema=schema)

  # Predict
  predictions = model.transform(input_data)

  # Write predictions
  predictions.select("state", "phone_number", "prediction", prob_will_churn('probability').alias("probability")) \
             .toJSON().foreach(writeln)


# Make predictions for each rdd
dstream.foreachRDD(make_predictions)

try:
  # Start stream processing
  ssc.start()
  ssc.awaitTermination()

except KeyboardInterrupt:
  # End stream processing
  ssc.stop(True, True)
  print("Exiting...")

# Done