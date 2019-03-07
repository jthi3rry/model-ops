#!/usr/bin/env python 
"""

  Example Usage:
  
    uwsgi --http 0.0.0.0:8080 --wsgi-file api_predict_churn.py --callable app 2>webapp.log &

  Example Request:

    curl -X POST http://localhost:8080/api/altus-ds-1/models/call-model \
         -H "Content-Type: application/json" \
         -H "Accept: application/json" \
         -d {
          "accessKey": "cdsw model secret key",
          "request": [
              {"state":"WV","account_length":141.0,"area_code":" 415","phone_number":" 330-8173","intl_plan":" yes","voice_mail_plan":" yes","number_vmail_messages":37.0,"total_day_minutes":258.6,"total_day_calls":84.0,"total_day_charge":43.96,"total_eve_minutes":222.0,"total_eve_calls":111.0,"total_eve_charge":18.87,"total_night_minutes":326.4,"total_night_calls":97.0,"total_night_charge":14.69,"total_intl_minutes":11.2,"total_intl_calls":5.0,"total_intl_charge":3.02,"number_customer_service_calls":0.0},
              {"state":"IN","account_length":65.0,"area_code":" 415","phone_number":" 329-6603","intl_plan":" no","voice_mail_plan":" no","number_vmail_messages":0.0,"total_day_minutes":129.1,"total_day_calls":137.0,"total_day_charge":21.95,"total_eve_minutes":228.5,"total_eve_calls":83.0,"total_eve_charge":19.42,"total_night_minutes":208.8,"total_night_calls":111.0,"total_night_charge":9.4,"total_intl_minutes":12.7,"total_intl_calls":6.0,"total_intl_charge":3.43,"number_customer_service_calls":4.0}
            ]
          }'

  Example Response:

    {
      "response": [
        {"churn": 0.0, "P_churn0": 0.83930023, "P_churn1": 0.16069977},
        {"churn": 1.0, "P_churn0": 0.28275055, "P_churn1": 0.71724945}
      ]
    }

"""
import os
import json
from flask import Flask, jsonify, request
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

master = os.getenv('MASTER', "local[*]")
model_path = os.getenv('MODEL_PATH', "hdfs:///tmp/churn.spark")

# Create Spark Session
spark = SparkSession.builder.master(master).getOrCreate()

# Declare schemas and UDFs
from pyspark.sql.types import *
from pyspark.sql.functions import udf
prob_will_churn = udf(lambda v: float(v[1]), FloatType())
prob_wont_churn = udf(lambda v: 1.0 - float(v[1]), FloatType())

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

# Define webapp
app = Flask(__name__)


@app.route('/')
def main():
    return jsonify({"api_name": "Simple Churn Prediction"})


@app.route('/api/altus-ds-1/models/call-model', methods=['POST'])
def predict():
  # Load data from request
  access_key = request.json.get("accessKey")
  data = request.json.get("request")

  # Create dataframe
  input_data = spark.sparkContext.parallelize(data).toDF(schema=schema)

  # Predict
  predictions = model.transform(input_data)

  # Get predictions as json
  output_data = {"response": list(map(json.loads,
    predictions.select(col('prediction').alias('churn'),
                       prob_will_churn('probability').alias("P_churn1"),
                       prob_wont_churn('probability').alias("P_churn0")) \
               .toJSON().collect()
  ))}

  # Return HTTP Response with predictions
  return jsonify(output_data)


if __name__ == '__main__':
  app.run(host="0.0.0.0")
