# MLOps from Zero with Databricks

## Logistics

* Six-Hour Weekly Sessions
* Use [Azure Open Datasets](https://docs.microsoft.com/en-us/azure/open-datasets/dataset-catalog#AzureDatabricks)

## Week by Week Schedule

### Week One:  Getting Started with Spark for MLOPs

#### Topics

* Topic:  Getting Started with Databricks and Spark
* Topic:  Getting Started with Continuous Integration

#### Learning Objectives

*   Compose Continuous Integration solutions
*   Compose Databricks/Spark Cluster Solutions

#### Readings/Media

* [Quickstart Azure Databricks](https://docs.microsoft.com/en-us/azure/databricks/scenarios/quickstart-create-databricks-workspace-portal?tabs=azure-portal)

##### Create tables directly from imported data

* [Introduction to importing, reading, and modifying data](https://docs.microsoft.com/en-us/azure/databricks/data/data)
* [Databases and tables](https://docs.microsoft.com/en-us/azure/databricks/data/tables)
* [Metastores](https://docs.microsoft.com/en-us/azure/databricks/data/metastores/)

##### Ingest data into Azure Databricks

* [Databricks integrations](https://docs.microsoft.com/en-us/azure/databricks/integrations/)

##### Access data in Apache Spark formats and from external data sources

* [Data sources](https://docs.microsoft.com/en-us/azure/databricks/data/data-sources/)
* [Delta Lake guide](https://docs.microsoft.com/en-us/azure/databricks/delta/)

#### Lab

* [Github Actions pytest](https://github.com/noahgift/github-actions-pytest)
* [quickstart-create-databricks-workspace-portal](https://docs.microsoft.com/azure/azure-databricks/quickstart-create-databricks-workspace-portal)

#### Discussions

* Why is CI (Continuous Integration) a foundational component for MLOps?
* Why is logging, monitoring and instrumentation so critical with distributed systems like Spark?

#### Assignments

* Build out your own Github repository with a Python scaffold of:  `Makefile`, `requirements.txt`, a source file, and a test file and do:  `make lint && make test && make format`.
* Add a Jupyter Notebook to your Continuous Integration setup and test it with `pytest --nbval`. [nbval plugin reference](https://github.com/computationalmodelling/nbval). 
* Perform Exploratory Data Analysis with a Databricks Spark Cluster using the [Azure Open Datasets](https://docs.microsoft.com/en-us/azure/open-datasets/dataset-catalog).

### Week Two:  Spark MLflow Tracking

#### Topics

* Topic:  Getting with MLflow tracking
* Topic:  Getting started with containerizing Microservices

#### Learning Objectives

*   Compose Spark MLflow tracking solutions
*   Compose Containerized Microservice solutions

#### Readings/Media

* [MLflow tracking](https://www.mlflow.org/docs/latest/tracking.html)
* [Databricks MLflow tracking](https://docs.databricks.com/applications/mlflow/tracking.html)
* [.NET 6 on AWS for Containers using Cloud9](https://github.com/noahgift/dot-net-6-aws)
* [Python MLOps Cookbook](https://github.com/noahgift/Python-MLOps-Cookbook)

#### Lab

#### Discussions

* How could experiment tracking increase productivity in a Data Science team?

#### Assignments

* Using an Azure Open Dataset perform several experiments using MLflow tracking and identify the best runs.
* Deploy a containerized application to a Cloud Provider or Local Datacenter.

### Week Three: Spark MLflow Projects

#### Topics

* Topic: Getting started with MLflow Projects
* Topic: Getting started with continuously deploying containerized Microservices
* Topic: Getting started with Kubernetes and Container Orchestration Solutions

#### Learning Objectives

*   Compose MLflow Projects solutions
*   Compose Containerized Microservice solutions

#### Readings/Media

#### Lab

#### Discussions

#### Assignments

* Run an MLflow Project on Databricks

### Week Four: Spark MLflow Models and Model Registry

#### Topics

* Topic: Getting started with MLflow Models
* Topic: Getting started with continuously deploying containerized Microservices

#### Learning Objectives

#### Readings/Media

* [MLflow Models](https://www.mlflow.org/docs/latest/models.html)
* [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)

#### Lab

#### Discussions

#### Assignments

* Evaluate a model with `mlflow.evaluate`:  https://www.mlflow.org/docs/latest/models.html#id20
