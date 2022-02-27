# Databricks notebook source
# MAGIC %md
# MAGIC ## Query databricks datasets

# COMMAND ----------

display(dbutils.fs.ls('/databricks-datasets'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get information about Databricks datasets

# COMMAND ----------

!head -n 5 /dbfs/databricks-datasets/README.md

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use shell command to query filesystem

# COMMAND ----------

!ls -l /dbfs/databricks-datasets/COVID

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create table via spark SQL

# COMMAND ----------

spark.sql("CREATE TABLE default.people10m OPTIONS (PATH 'dbfs:/databricks-datasets/learning-spark-v2/people/people-10m.delta')")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Table in SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from people10m

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query data in Python

# COMMAND ----------

people = spark.sql("select * from people10m")
display(people.select("*"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Describe Table Price Column

# COMMAND ----------

people.describe(['salary']).show()

# COMMAND ----------


