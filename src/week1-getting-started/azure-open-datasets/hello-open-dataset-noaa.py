# Databricks notebook source
# MAGIC %md
# MAGIC ## Install and import as shown
# MAGIC https://github.com/Azure/OpenDatasetsNotebooks/blob/master/tutorials/data-access/01-weather-to-spark-dataframe.ipynb

# COMMAND ----------

from azureml.opendatasets import NoaaIsdWeather
from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta

# COMMAND ----------

# MAGIC %md
# MAGIC Set date range

# COMMAND ----------

start_date = parser.parse('2019-1-1')
end_date = parser.parse('2019-3-31')
isd = NoaaIsdWeather(start_date, end_date)
df = isd.to_spark_dataframe()
display(df.limit(10))

# COMMAND ----------


