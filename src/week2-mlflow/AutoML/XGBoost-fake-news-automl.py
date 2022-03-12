# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **10-3-ML-Cluster** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/406583024052808/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/406583024052798) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC 
# MAGIC Runtime Version: _10.3.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow
import databricks.automl_runtime

# Use MLflow to track experiments
mlflow.set_experiment("/Users/noah.gift@gmail.com/databricks_automl/label_news_articles_csv-2022_03_12-15_38")

target_col = "label"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

from mlflow.tracking import MlflowClient
import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_client = MlflowClient()
input_data_path = input_client.download_artifacts("c2dfe80b419d4a8dbc88a90e3274369a", "data", input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

df_loaded.head(1).to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["text_without_stopwords", "published", "language", "main_img_url", "site_url", "hasImage", "title_without_stopwords", "text", "title", "type", "author"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(handle_unknown="ignore")

transformers.append(("onehot", one_hot_encoder, ["published", "language", "site_url", "hasImage", "title", "title_without_stopwords", "text_without_stopwords"]))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Medium-cardinality categoricals
# MAGIC Convert each medium-cardinality categorical column into a numerical representation.
# MAGIC Each string column is hashed to 1024 float columns.
# MAGIC Each numeric column is imputed with zeros.

# COMMAND ----------

from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

for feature in ["text", "main_img_url"]:
    hash_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
        (f"{feature}_hasher", FeatureHasher(n_features=1024, input_type="string"))])
    transformers.append((f"{feature}_hasher", hash_transformer, [feature]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text features
# MAGIC Convert each feature to a fixed-length vector using TF-IDF vectorization. The length of the output
# MAGIC vector is equal to 1024. Each column corresponds to one of the top word n-grams
# MAGIC where n is in the range [1, 2].

# COMMAND ----------

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

for col in {'type', 'author'}:
    vectorizer = Pipeline(steps=[
        ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
        # Reshape to 1D since SimpleImputer changes the shape of the input to 2D
        ("reshape", FunctionTransformer(np.reshape, kw_args={"newshape":-1})),
        ("tfidf", TfidfVectorizer(decode_error="ignore", ngram_range = (1, 2), max_features=1024))])

    transformers.append((f"text_{col}", vectorizer, [col]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature standardization
# MAGIC Scale all feature columns to be centered around zero with unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

df_loaded.columns

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

# Split out train data
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, random_state=799811440, stratify=split_y)

# Split remaining data equally for validation and test
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, random_state=799811440, stratify=split_y_rem)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/406583024052808/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBClassifier

help(XGBClassifier)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display="diagram")

xgbc_classifier = XGBClassifier(
  colsample_bytree=0.7324555878929649,
  learning_rate=0.007636627530856404,
  max_depth=7,
  min_child_weight=6,
  n_estimators=106,
  n_jobs=100,
  subsample=0.6972187716458148,
  verbosity=0,
  random_state=799811440,
)

model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", xgbc_classifier),
])

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
])

mlflow.sklearn.autolog(disable=True)
X_val_processed = pipeline.fit_transform(X_val, y_val)

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="xgboost") as mlflow_run:
    model.fit(X_train, y_train, classifier__early_stopping_rounds=5, classifier__eval_set=[(X_val_processed,y_val)], classifier__verbose=False)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    xgbc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    xgbc_val_metrics = {k.replace("val_", ""): v for k, v in xgbc_val_metrics.items()}
    xgbc_test_metrics = {k.replace("test_", ""): v for k, v in xgbc_test_metrics.items()}
    display(pd.DataFrame([xgbc_val_metrics, xgbc_test_metrics], index=["validation", "test"]))

# COMMAND ----------

# Patch requisite packages to the model environment YAML for model serving
import os
import shutil
import uuid
import yaml

None

import xgboost
from mlflow.tracking import MlflowClient

xgbc_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(xgbc_temp_dir)
xgbc_client = MlflowClient()
xgbc_model_env_path = xgbc_client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", xgbc_temp_dir)
xgbc_model_env_str = open(xgbc_model_env_path)
xgbc_parsed_model_env_str = yaml.load(xgbc_model_env_str, Loader=yaml.FullLoader)

xgbc_parsed_model_env_str["dependencies"][-1]["pip"].append(f"xgboost=={xgboost.__version__}")

with open(xgbc_model_env_path, "w") as f:
  f.write(yaml.dump(xgbc_parsed_model_env_str))
xgbc_client.log_artifact(run_id=mlflow_run.info.run_id, local_path=xgbc_model_env_path, artifact_path="model")
shutil.rmtree(xgbc_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = True

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # SHAP cannot explain models using data with nulls.
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).
    mode = X_train.mode().iloc[0]

    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=1).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="logit")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading model to make prediction

# COMMAND ----------

model_uri = f"runs:/51c0348482e042ea8e4b7983ab6bff99/model"

model = mlflow.pyfunc.load_model(model_uri)
#model.predict(input_X)

# COMMAND ----------

import pandas as pd
data = {'author': {0: 'bigjim.com'},
 'published': {0: '2016-10-27T18:05:26.351+03:00'},
 'title': {0: 'aliens are coming to invade earth'},
 'text': {0: 'aliens are coming to invade earth'},
 'language': {0: 'english'},
 'site_url': {0: 'cnn.com'},
 'main_img_url': {0: 'https://2.bp.blogspot.com/-0mdp0nZiwMI/UYwYvexmW2I/AAAAAAAAVQM/7C_X5WRE_mQ/w1200-h630-p-nu/Edison-Stock-Ticker.jpg'},
 'type': {0: 'bs'},
 'title_without_stopwords': {0: 'aliens are coming to invade earth'},
 'text_without_stopwords': {0: 'aliens are coming to invade earth'},
 'hasImage': {0: 1.0}}
df = pd.DataFrame(data=data)
df.head()

# COMMAND ----------

model.predict(df)

# COMMAND ----------


