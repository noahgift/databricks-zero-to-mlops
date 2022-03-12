# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC This notebook performs exploratory data analysis on the dataset.
# MAGIC To expand on the analysis, attach this notebook to the **10-3-ML-Cluster** cluster,
# MAGIC edit [the options of pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/advanced_usage.html), and rerun it.
# MAGIC - Explore completed trials in the [MLflow experiment](#mlflow/experiments/1491276346262250/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/1491276346262240) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC 
# MAGIC Runtime Version: _10.3.x-cpu-ml-scala2.12_

# COMMAND ----------

import os
import uuid
import shutil
import pandas as pd
import databricks.automl_runtime

from mlflow.tracking import MlflowClient

# Download input data from mlflow into a pandas DataFrame
# Create temporary directory to download data
temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(temp_dir)

# Download the artifact and read it
client = MlflowClient()
training_data_path = client.download_artifacts("0c1c66aab3d940ec9d8a0c91b1fab811", "data", temp_dir)
df = pd.read_parquet(os.path.join(training_data_path, "training_data"))

# Delete the temporary data
shutil.rmtree(temp_dir)

target_col = "POSITION"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Results

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(df, minimal=True, title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)
