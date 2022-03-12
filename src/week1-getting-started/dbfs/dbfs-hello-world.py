# Databricks notebook source
# MAGIC %md
# MAGIC ## Create directory and file on DBFS

# COMMAND ----------

# MAGIC %fs ls file:/tmp

# COMMAND ----------

# MAGIC %fs mkdirs file:/tmp/my_local_dir

# COMMAND ----------

# MAGIC %fs ls file:/tmp/my_local_dir

# COMMAND ----------

dbutils.fs.put("file:/tmp/my_local_dir/my_new_file", "This is a file on the local driver node.")

# COMMAND ----------

# MAGIC %fs ls file:/tmp/my_local_dir

# COMMAND ----------


