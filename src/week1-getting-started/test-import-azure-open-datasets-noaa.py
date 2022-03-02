# Databricks notebook source
from azureml.opendatasets import NoaaIsdWeather

from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta

# COMMAND ----------

start_date = parser.parse('2019-1-1')
end_date = parser.parse('2019-3-31')
isd = NoaaIsdWeather(start_date, end_date)
df = isd.to_spark_dataframe()
display(df.limit(10))

# COMMAND ----------


