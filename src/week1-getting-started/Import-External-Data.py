# Databricks notebook source
# MAGIC %md
# MAGIC ### Load imported data

# COMMAND ----------

df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/noah.gift@gmail.com/nba-1.csv")

# COMMAND ----------

display(df1)

# COMMAND ----------


