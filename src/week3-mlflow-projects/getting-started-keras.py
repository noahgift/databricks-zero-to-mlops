# Databricks notebook source
# MAGIC %md 
# MAGIC # Getting started with deep learning in Databricks: an end-to-end example using TensorFlow Keras, Hyperopt, and MLflow
# MAGIC 
# MAGIC This tutorial uses a small dataset to show how to use TensorFlow Keras, Hyperopt, and MLflow to develop a deep learning model in Databricks. 
# MAGIC 
# MAGIC It includes the following steps:
# MAGIC - Load and preprocess data
# MAGIC - Part 1. Create a neural network model with TensorFlow Keras and view training with inline TensorBoard
# MAGIC - Part 2. Perform automated hyperparameter tuning with Hyperopt and MLflow and use autologging to save results
# MAGIC - Part 3. Use the best set of hyperparameters to build a final model 
# MAGIC - Part 4. Register the model in MLflow and use the model to make predictions
# MAGIC 
# MAGIC ### Setup
# MAGIC - Databricks Runtime for Machine Learning 7.0 or above. This notebook uses TensorBoard to display the results of neural network training. Depending on the version of Databricks Runtime you are using, you use different methods to start TensorBoard. 

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow

# COMMAND ----------

# MAGIC %md ## Load and preprocess data
# MAGIC This example uses the California Housing dataset from `scikit-learn`. 

# COMMAND ----------

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

cal_housing = fetch_california_housing()

# Split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2)

# COMMAND ----------

# MAGIC %md ### Scale features
# MAGIC Feature scaling is important when working with neural networks. This notebook uses the `scikit-learn` function `StandardScaler`.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md ## Part 1. Create model and view TensorBoard in notebook

# COMMAND ----------

# MAGIC %md ### Create the neural network

# COMMAND ----------

def create_model():
  model = Sequential()
  model.add(Dense(20, input_dim=8, activation="relu"))
  model.add(Dense(20, activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

# MAGIC %md ### Compile the model

# COMMAND ----------

model = create_model()

model.compile(loss="mse",
              optimizer="Adam",
              metrics=["mse"])

# COMMAND ----------

# MAGIC %md ### Create callbacks

# COMMAND ----------

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# In the following lines, replace <username> with your username.
experiment_log_dir = "/dbfs/<username>/tb"
checkpoint_path = "/dbfs/<username>/keras_checkpoint_weights.ckpt"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3)

history = model.fit(X_train, y_train, validation_split=.2, epochs=35, callbacks=[tensorboard_callback, model_checkpoint, early_stopping])

# COMMAND ----------

# MAGIC %md ### TensorBoard commands for Databricks Runtime 7.2 ML and above
# MAGIC 
# MAGIC When you start TensorBoard this way, it continues to run until you detach the notebook from the cluster.  
# MAGIC Note: to clear the TensorBoard between runs, use this command: `dbutils.fs.rm(experiment_log_dir.replace("/dbfs",""), recurse=True)`

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------

# MAGIC %md ### TensorBoard commands for Databricks Runtime 7.1 ML and below
# MAGIC 
# MAGIC The command in the following cell displays a link that, when clicked, opens TensorBoard in a new tab.
# MAGIC 
# MAGIC When you start TensorBoard this way, it continues to run until you either stop it with `dbutils.tensorboard.stop()` or you shut down the cluster.

# COMMAND ----------

#dbutils.tensorboard.start(experiment_log_dir)

# COMMAND ----------

# MAGIC %md ### Evaluate model on test dataset

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

# MAGIC %md ## Part 2. Hyperparameter tuning with Hyperopt and MLflow
# MAGIC [Hyperopt](https://github.com/hyperopt/hyperopt) is a Python library for hyperparameter tuning. Databricks Runtime for Machine Learning includes an optimized and enhanced version of Hyperopt, including automated MLflow tracking. For more information about using Hyperopt, see the [Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin).

# COMMAND ----------

# MAGIC %md ### Create neural network model using variables for number of nodes in hidden layers

# COMMAND ----------

def create_model(n):
  model = Sequential()
  model.add(Dense(int(n["dense_l1"]), input_dim=8, activation="relu"))
  model.add(Dense(int(n["dense_l2"]), activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

# MAGIC %md ### Create Hyperopt objective function

# COMMAND ----------

from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

def runNN(n):
  # Import tensorflow 
  import tensorflow as tf
  
  # Log run information with mlflow.tensorflow.autolog()
  mlflow.tensorflow.autolog()
  
  model = create_model(n)

  # Select optimizer
  optimizer_call = getattr(tf.keras.optimizers, n["optimizer"])
  optimizer = optimizer_call(learning_rate=n["learning_rate"])
 
  # Compile model
  model.compile(loss="mse",
                optimizer=optimizer,
                metrics=["mse"])

  history = model.fit(X_train, y_train, validation_split=.2, epochs=10, verbose=2)

  # Evaluate the model
  score = model.evaluate(X_test, y_test, verbose=0)
  obj_metric = score[0]  
  return {"loss": obj_metric, "status": STATUS_OK}

# COMMAND ----------

# MAGIC %md ### Define Hyperopt search space

# COMMAND ----------

space = {
  "dense_l1": hp.quniform("dense_l1", 10, 30, 1),
  "dense_l2": hp.quniform("dense_l2", 10, 30, 1),
  "learning_rate": hp.loguniform("learning_rate", -5, 0),
  "optimizer": hp.choice("optimizer", ["Adadelta", "Adam"])
 }

# COMMAND ----------

# MAGIC %md ### Create the `SparkTrials` object
# MAGIC 
# MAGIC The `SparkTrials` object tells `fmin()` to distribute the tuning job across a Spark cluster. When you create the `SparkTrials` object, you can use the `parallelism` argument to set the maximum number of trials to evaluate concurently. The default setting is the number of Spark executors available.  
# MAGIC 
# MAGIC A higher number lets you scale-out testing of more hyperparameter settings. Because Hyperopt proposes new trials based on past results, there is a trade-off between parallelism and adaptivity. For a fixed `max_evals`, greater parallelism speeds up calculations, but lower parallelism may lead to better results since each iteration has access to more past results.

# COMMAND ----------

# If you do not specify a parallelism argument, the default is the number of available Spark executors 
spark_trials = SparkTrials()

# COMMAND ----------

# MAGIC %md ### Perform hyperparameter tuning 
# MAGIC Put the `fmin()` call inside an MLflow run to save results to MLflow. MLflow tracks the parameters and performance metrics of each run.   
# MAGIC 
# MAGIC After running the following cell, you can view the results in MLflow. Click **Experiment** at the upper right to display the Experiment Runs sidebar. Click the icon at the far right next to **Experiment Runs** to display the MLflow Runs Table.
# MAGIC 
# MAGIC For more information about using MLflow to analyze runs, see ([AWS](https://docs.databricks.com/applications/mlflow/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/index.html)).

# COMMAND ----------

with mlflow.start_run():
  best_hyperparam = fmin(fn=runNN, 
                         space=space, 
                         algo=tpe.suggest, 
                         max_evals=30, 
                         trials=spark_trials)

# COMMAND ----------

# MAGIC %md ## Part 3. Use the best set of hyperparameters to build a final model

# COMMAND ----------

import hyperopt

print(hyperopt.space_eval(space, best_hyperparam))

# COMMAND ----------

first_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l1"]
second_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l2"]
learning_rate = hyperopt.space_eval(space, best_hyperparam)["learning_rate"]
optimizer = hyperopt.space_eval(space, best_hyperparam)["optimizer"]

# COMMAND ----------

# Get optimizer and update with learning_rate value
optimizer_call = getattr(tf.keras.optimizers, optimizer)
optimizer = optimizer_call(learning_rate=learning_rate)

# COMMAND ----------

def create_new_model():
  model = Sequential()
  model.add(Dense(first_layer, input_dim=8, activation="relu"))
  model.add(Dense(second_layer, activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

new_model = create_new_model()
  
new_model.compile(loss="mse",
                optimizer=optimizer,
                metrics=["mse"])

# COMMAND ----------

# MAGIC %md When `autolog()` is active, MLflow does not automatically end a run. We need to end the run that was started in Cmd 30 before starting and autologging a new run.  
# MAGIC For more information, see https://www.mlflow.org/docs/latest/tracking.html#automatic-logging.

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

import matplotlib.pyplot as plt

mlflow.tensorflow.autolog()

with mlflow.start_run() as run:
  
  history = new_model.fit(X_train, y_train, epochs=35, callbacks=[early_stopping])
  
  # Save the run information to register the model later
  kerasURI = run.info.artifact_uri
  
  # Evaluate model on test dataset and log result
  mlflow.log_param("eval_result", new_model.evaluate(X_test, y_test)[0])
  
  # Plot predicted vs known values for a quick visual check of the model and log the plot as an artifact
  keras_pred = new_model.predict(X_test)
  plt.plot(y_test, keras_pred, "o", markersize=2)
  plt.xlabel("observed value")
  plt.ylabel("predicted value")
  plt.savefig("kplot.png")
  mlflow.log_artifact("kplot.png") 

# COMMAND ----------

# MAGIC %md ## Part 4. Register the model in MLflow and use the model to make predictions
# MAGIC To learn more about the Model Registry, see ([AWS](https://docs.databricks.com/applications/mlflow/model-registry.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-registry)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/model-registry.html)).

# COMMAND ----------

import time

model_name = "cal_housing_keras"
model_uri = kerasURI+"/model"
new_model_version = mlflow.register_model(model_uri, model_name)

# Registering the model takes a few seconds, so add a delay before continuing with the next cell
time.sleep(5)

# COMMAND ----------

# MAGIC %md ### Load the model for inference and make predictions

# COMMAND ----------

keras_model = mlflow.keras.load_model(f"models:/{model_name}/{new_model_version.version}")

keras_pred = keras_model.predict(X_test)
keras_pred

# COMMAND ----------

# MAGIC %md ## Clean up
# MAGIC To stop TensorBoard:
# MAGIC - If you are running Databricks Runtime for Machine Learning 7.1 ML or below, uncomment and run the command in the following cell.  
# MAGIC - If you are running Databricks Runtime for Machine Learning 7.2 ML or above, detach this notebook from the cluster.

# COMMAND ----------

#dbutils.tensorboard.stop()
