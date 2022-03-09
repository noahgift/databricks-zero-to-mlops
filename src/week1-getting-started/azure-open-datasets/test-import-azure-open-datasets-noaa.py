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

#upload more datasets (do tab complete)
#from azureml.opendatasets import 

# COMMAND ----------

import numpy as np
import pyspark.pandas as ps

# COMMAND ----------

psdf = ps.DataFrame(df)

# COMMAND ----------

psdf.describe()

# COMMAND ----------

display(psdf)

# COMMAND ----------


