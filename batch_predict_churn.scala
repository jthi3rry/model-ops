import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions.{udf,col}
import org.apache.spark.ml.linalg.DenseVector

val model_path = scala.util.Properties.envOrElse("MODEL", "hdfs:///tmp/churn.spark")
val input_path = scala.util.Properties.envOrElse("INPUT_PATH", "hdfs:///tmp/churn.all")
val output_table = scala.util.Properties.envOrElse("OUTPUT_TABLE", "thierry.churn_predictions")

// Declare schemas and UDFs
val prob_will_churn = udf{ (x:DenseVector) => x(1) }

val schema = new StructType().
  add("state", "string").
  add("account_length", "double").
  add("area_code", "string").
  add("phone_number", "string").
  add("intl_plan", "string").
  add("voice_mail_plan", "string").
  add("number_vmail_messages", "double").
  add("total_day_minutes", "double").
  add("total_day_calls", "double").
  add("total_day_charge", "double").
  add("total_eve_minutes", "double").
  add("total_eve_calls", "double").
  add("total_eve_charge", "double").
  add("total_night_minutes", "double").
  add("total_night_calls", "double").
  add("total_night_charge", "double").
  add("total_intl_minutes", "double").
  add("total_intl_calls", "double").
  add("total_intl_charge", "double").
  add("number_customer_service_calls", "double")
  // add("churned", "string")

// Restore model
val model = PipelineModel.load("hdfs:///tmp/churn.spark")

// Load data from file
val input_data = spark.read.schema(schema).option("header", false).csv(input_path)

// Predict  
val predictions = model.transform(input_data)

// Write predictions to table
predictions.select(col("state"), col("phone_number"), col("prediction"), prob_will_churn(col("probability")).alias("probability")).
            write.mode("overwrite").saveAsTable("thierry.churn_predictions_2")

// Done
