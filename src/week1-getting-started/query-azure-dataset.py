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
