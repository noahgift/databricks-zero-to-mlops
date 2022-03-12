# Databricks notebook source
# MAGIC %md # Databricks ML Quickstart: Model Training
# MAGIC 
# MAGIC This notebook provides a quick overview of machine learning model training on Databricks. To train models, you can use libraries like scikit-learn that are preinstalled on the Databricks Runtime for Machine Learning. In addition, you can use MLflow to track the trained models, and Hyperopt with SparkTrials to scale hyperparameter tuning.
# MAGIC 
# MAGIC This tutorial covers:
# MAGIC - Part 1: Training a simple classification model with MLflow tracking
# MAGIC - Part 2: Hyperparameter tuning a better performing model with Hyperopt
# MAGIC 
# MAGIC For more details on productionizing machine learning on Databricks including model lifecycle management and model inference, see the ML End to End Example ([AWS](https://docs.databricks.com/applications/mlflow/end-to-end-example.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/end-to-end-example)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/end-to-end-example.html)).
# MAGIC 
# MAGIC ### Requirements
# MAGIC - Cluster running Databricks Runtime 7.5 ML or above

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Libraries
# MAGIC Import the necessary libraries. These libraries are preinstalled on Databricks Runtime for Machine Learning ([AWS](https://docs.databricks.com/runtime/mlruntime.html)|[Azure](https://docs.microsoft.com/azure/databricks/runtime/mlruntime)|[GCP](https://docs.gcp.databricks.com/runtime/mlruntime.html)) clusters and are tuned for compatibility and performance.

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load data
# MAGIC The tutorial uses a dataset describing different wine samples. The [dataset](https://archive.ics.uci.edu/ml/datasets/Wine) is from the UCI Machine Learning Repository and is included in DBFS ([AWS](https://docs.databricks.com/data/databricks-file-system.html)|[Azure](https://docs.microsoft.com/azure/databricks/data/databricks-file-system)|[GCP](https://docs.gcp.databricks.com/data/databricks-file-system.html)).
# MAGIC The goal is to classify red and white wines by their quality. 
# MAGIC 
# MAGIC For more details on uploading and loading from other data sources, see the documentation on working with data ([AWS](https://docs.databricks.com/data/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/data/index)|[GCP](https://docs.gcp.databricks.com/data/index.html)).

# COMMAND ----------

# Load and preprocess data
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=';')
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=';')
white_wine['is_red'] = 0.0
red_wine['is_red'] = 1.0
data_df = pd.concat([white_wine, red_wine], axis=0)

# Define classification labels based on the wine quality
data_labels = data_df['quality'] >= 7
data_df = data_df.drop(['quality'], axis=1)

# Split 80/20 train-test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
  data_df,
  data_labels,
  test_size=0.2,
  random_state=1
)

# COMMAND ----------

# MAGIC %md ## Part 1. Train a classification model

# COMMAND ----------

# MAGIC %md ### MLflow Tracking
# MAGIC [MLflow tracking](https://www.mlflow.org/docs/latest/tracking.html) allows you to organize your machine learning training code, parameters, and models. 
# MAGIC 
# MAGIC You can enable automatic MLflow tracking by using [*autologging*](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging).

# COMMAND ----------

# Enable MLflow autologging for this notebook
mlflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC Next, train a classifier within the context of an MLflow run, which automatically logs the trained model and many associated metrics and parameters. 
# MAGIC 
# MAGIC You can supplement the logging with additional metrics such as the model's AUC score on the test dataset.

# COMMAND ----------

with mlflow.start_run(run_name='gradient_boost') as run:
  model = sklearn.ensemble.GradientBoostingClassifier(random_state=0)
  
  # Models, parameters, and training metrics are tracked automatically
  model.fit(X_train, y_train)

  predicted_probs = model.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  
  # The AUC score on test data is not automatically logged, so log it manually
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC If you aren't happy with the performance of this model, train another model with different hyperparameters.

# COMMAND ----------

# Start a new run and assign a run_name for future reference
with mlflow.start_run(run_name='gradient_boost') as run:
  model_2 = sklearn.ensemble.GradientBoostingClassifier(
    random_state=0, 
    
    # Try a new parameter setting for n_estimators
    n_estimators=200,
  )
  model_2.fit(X_train, y_train)

  predicted_probs = model_2.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md ### View MLflow runs
# MAGIC To view the logged training runs, click the **Experiment** icon at the upper right of the notebook to display the experiment sidebar. If necessary, click the refresh icon to fetch and monitor the latest runs. 
# MAGIC 
# MAGIC <img width="350" src="https://docs.databricks.com/_static/images/mlflow/quickstart/experiment-sidebar-icons.png"/>
# MAGIC 
# MAGIC You can then click the experiment page icon to display the more detailed MLflow experiment page ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#notebook-experiments)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#notebook-experiments)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#notebook-experiments)). This page allows you to compare runs and view details for specific runs.
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/compare-runs.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load models
# MAGIC You can also access the results for a specific run using the MLflow API. The code in the following cell illustrates how to load the model trained in a given MLflow run and use it to make predictions. You can also find code snippets for loading specific models on the MLflow run page ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)).

# COMMAND ----------

# After a model has been logged, you can load it in different notebooks or jobs
# mlflow.pyfunc.load_model makes model prediction available under a common API
model_loaded = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=run.info.run_id
  )
)

predictions_loaded = model_loaded.predict(X_test)
predictions_original = model_2.predict(X_test)

# The loaded model should match the original
assert(np.array_equal(predictions_loaded, predictions_original))

# COMMAND ----------

# MAGIC %md ## Part 2. Hyperparameter Tuning
# MAGIC At this point, you have trained a simple model and used the MLflow tracking service to organize your work. This section covers how to perform more sophisticated tuning using Hyperopt.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parallel training with Hyperopt and SparkTrials
# MAGIC [Hyperopt](http://hyperopt.github.io/hyperopt/) is a Python library for hyperparameter tuning. For more information about using Hyperopt in Databricks, see the documentation ([AWS](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/automl-hyperparam-tuning/index#hyperparameter-tuning-with-hyperopt)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)).
# MAGIC 
# MAGIC You can use Hyperopt with SparkTrials to run hyperparameter sweeps and train multiple models in parallel. This reduces the time required to optimize model performance. MLflow tracking is integrated with Hyperopt to automatically log models and parameters.

# COMMAND ----------

# Define the search space to explore
search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 1000, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
}

def train_model(params):
  # Enable autologging on each worker
  mlflow.autolog()
  with mlflow.start_run(nested=True):
    model_hp = sklearn.ensemble.GradientBoostingClassifier(
      random_state=0,
      **params
    )
    model_hp.fit(X_train, y_train)
    predicted_probs = model_hp.predict_proba(X_test)
    # Tune based on the test AUC
    # In production settings, you could use a separate validation set instead
    roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
    mlflow.log_metric('test_auc', roc_auc)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*roc_auc}

# SparkTrials distributes the tuning using Spark workers
# Greater parallelism speeds processing, but each hyperparameter trial has less information from other trials
# On smaller clusters or Databricks Community Edition try setting parallelism=2
spark_trials = SparkTrials(
  parallelism=8
)

with mlflow.start_run(run_name='gb_hyperopt') as run:
  # Use hyperopt to find the parameters yielding the highest AUC
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=32,
    trials=spark_trials)

# COMMAND ----------

# MAGIC %md ### Search runs to retrieve the best model
# MAGIC Because all of the runs are tracked by MLflow, you can retrieve the metrics and parameters for the best run using the MLflow search runs API to find the tuning run with the highest test auc.
# MAGIC 
# MAGIC This tuned model should perform better than the simpler models trained in Part 1. 

# COMMAND ----------

# Sort runs by their test auc; in case of ties, use the most recent run
best_run = mlflow.search_runs(
  order_by=['metrics.test_auc DESC', 'start_time DESC'],
  max_results=10,
).iloc[0]
print('Best Run')
print('AUC: {}'.format(best_run["metrics.test_auc"]))
print('Num Estimators: {}'.format(best_run["params.n_estimators"]))
print('Max Depth: {}'.format(best_run["params.max_depth"]))
print('Learning Rate: {}'.format(best_run["params.learning_rate"]))

best_model_pyfunc = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=best_run.run_id
  )
)
best_model_predictions = best_model_pyfunc.predict(X_test[:5])
print("Test Predictions: {}".format(best_model_predictions))

# COMMAND ----------

# MAGIC %md ### Compare multiple runs in the UI
# MAGIC As in Part 1, you can view and compare the runs in the MLflow experiment details page, accessible via the external link icon at the top of the **Experiment** sidebar. 
# MAGIC 
# MAGIC On the experiment details page, click the "+" icon to expand the parent run, then select all runs except the parent, and click **Compare**. You can visualize the different runs using a parallel coordinates plot, which shows the impact of different parameter values on a metric. 
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/parallel-plot.png"/>
