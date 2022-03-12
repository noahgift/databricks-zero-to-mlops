# Databricks notebook source
from azureml.opendatasets import NycTlcGreen
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pyspark.pandas as ps

# COMMAND ----------

green_taxi_df = ps.DataFrame([])
start = datetime.strptime("1/1/2015","%m/%d/%Y")
end = datetime.strptime("1/31/2015","%m/%d/%Y")

for sample_month in range(12):
    green_taxi_df = NycTlcGreen(start + relativedelta(months=sample_month), end + relativedelta(months=sample_month)) \
        .to_pandas_dataframe()

green_taxi_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC remove columns you don't need

# COMMAND ----------

columns_to_remove = ["lpepDropoffDatetime", "puLocationId", "doLocationId", "extra", "mtaTax",
                     "improvementSurcharge", "tollsAmount", "ehailFee", "tripType", "rateCodeID",
                     "storeAndFwdFlag", "paymentType", "fareAmount", "tipAmount"
                    ]
for col in columns_to_remove:
    green_taxi_df.pop(col)

green_taxi_df.head(5)

# COMMAND ----------

display(green_taxi_df)

# COMMAND ----------

# MAGIC %md
# MAGIC filter out anomolies

# COMMAND ----------

final_df = green_taxi_df.query("pickupLatitude>=40.53 and pickupLatitude<=40.88")
final_df = final_df.query("pickupLongitude>=-74.09 and pickupLongitude<=-73.72")
final_df = final_df.query("tripDistance>=0.25 and tripDistance<31")
final_df = final_df.query("passengerCount>0 and totalAmount>0")

columns_to_remove_for_training = ["pickupLongitude", "pickupLatitude", "dropoffLongitude", "dropoffLatitude"]
for col in columns_to_remove_for_training:
    final_df.pop(col)

# COMMAND ----------

display(final_df)

# COMMAND ----------


