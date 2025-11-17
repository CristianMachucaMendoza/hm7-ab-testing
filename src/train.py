""""
Description: PySpark script for training a fraud detection model.
"""

import os
import sys
import traceback
import argparse
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
import time
from datetime import datetime

# pylint: disable=broad-exception-caught

def create_spark_session(s3_config=None):
    """Create and configure a Spark session."""
    print("DEBUG: Начинаем создание Spark сессии")
    try:
        builder = (SparkSession
            .builder
            .appName("FraudDetectionModel")
        )

        if s3_config and all(k in s3_config for k in ['endpoint_url', 'access_key', 'secret_key']):
            print(f"DEBUG: Настраиваем S3 с endpoint_url: {s3_config['endpoint_url']}")
            builder = (builder
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.endpoint", s3_config['endpoint_url'])
                .config("spark.hadoop.fs.s3a.access.key", s3_config['access_key'])
                .config("spark.hadoop.fs.s3a.secret.key", s3_config['secret_key'])
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
            )

        print("DEBUG: Spark сессия успешно сконфигурирована")
        return builder
    except Exception as e:
        print(f"ERROR: Ошибка создания Spark сессии: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def load_data(spark, input_path):
    """Load and prepare the fraud detection dataset."""
    print(f"DEBUG: Начинаем загрузку данных из: {input_path}")
    try:
        print(f"DEBUG: Чтение parquet файла из {input_path}")
        df = spark.read.parquet(input_path, header=True, inferSchema=True)

        print("Dataset Schema:")
        df.printSchema()
        print(f"Total records: {df.count()}")

        print("DEBUG: Первые 5 строк данных:")
        df.show(5, truncate=False)

        print("DEBUG: Разделение на обучающую и тестовую выборки")
        train_df, test_df = df.randomSplit([0.005, 0.0001], seed=42)
        print(f"Training set size: {train_df.count()}")
        print(f"Testing set size: {test_df.count()}")

        return train_df, test_df
    except Exception as e:
        print(f"ERROR: Ошибка загрузки данных: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def prepare_features(train_df, test_df):
    """Prepare features for model training."""
    print("DEBUG: Начинаем подготовку признаков")
    try:
        dtypes = dict(train_df.dtypes)
        print(f"DEBUG: Типы данных: {dtypes}")

        feature_cols = [col for col in train_df.columns
                        if col not in ('tx_fraud', 'tx_timestamp') and dtypes[col] != 'string']
        print(f"DEBUG: Выбрано {len(feature_cols)} числовых признаков: {feature_cols}")

        print("DEBUG: Проверка наличия нулевых значений в обучающей выборке")
        for col in train_df.columns:
            null_count = train_df.filter(train_df[col].isNull()).count()
            if null_count > 0:
                print(f"WARNING: Колонка '{col}' содержит {null_count} нулевых значений")

        return train_df, test_df, feature_cols
    except Exception as e:
        print(f"ERROR: Ошибка подготовки признаков: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def train_model(train_df, test_df, feature_cols, model_type="rf", run_name="fraud_detection_model"):
    """Train a fraud detection model and log metrics to MLflow."""
    print(f"DEBUG: Starting training model type {model_type}, run_name: {run_name}")
    try:
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        classifier = RandomForestClassifier(
            labelCol="tx_fraud",
            featuresCol="features",
            numTrees=10,
            maxDepth=5
        )

        pipeline = Pipeline(stages=[assembler, scaler, classifier])

        evaluator_auc = BinaryClassificationEvaluator(
            labelCol="tx_fraud",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="tx_fraud",
            predictionCol="prediction",
            metricName="accuracy"
        )
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="tx_fraud",
            predictionCol="prediction",
            metricName="f1"
        )

        print(f"DEBUG: Starting MLflow run: {run_name}")
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"MLflow Run ID: {run_id}")

            mlflow.log_param("numTrees", 10)
            mlflow.log_param("maxDepth", 5)

            print("DEBUG: Training the model...")
            model = pipeline.fit(train_df)
            print("DEBUG: Model trained successfully")

            predictions = model.transform(test_df)
            print("DEBUG: Predictions done")

            auc = evaluator_auc.evaluate(predictions)
            accuracy = evaluator_acc.evaluate(predictions)
            f1 = evaluator_f1.evaluate(predictions)

            print("DEBUG: Logging metrics to MLflow")
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)

            rf_model = model.stages[-1]
            try:
                num_trees = rf_model.getNumTrees
                max_depth = rf_model.getMaxDepth()
                print(f"DEBUG: numTrees={num_trees}, maxDepth={max_depth}")
                mlflow.log_param("best_numTrees", num_trees)
                mlflow.log_param("best_maxDepth", max_depth)
            except Exception as e:
                print(f"WARNING: Error getting model params: {str(e)}")

            print("DEBUG: Logging the model to MLflow")
            mlflow.spark.log_model(model, "model")

            print(f"AUC: {auc}")
            print(f"Accuracy: {accuracy}")
            print(f"F1 Score: {f1}")

            metrics = {
                "run_id": run_id,
                "auc": auc,
                "accuracy": accuracy,
                "f1": f1
            }

            return model, metrics

    except Exception as e:
        print(f"ERROR: Error training model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def save_model(model, output_path):
    """Save the trained model to the specified path."""
    print(f"DEBUG: Сохраняем модель в: {output_path}")
    try:
        model.write().overwrite().save(output_path)
        print(f"Model saved to: {output_path}")
    except Exception as e:
        print(f"ERROR: Ошибка сохранения модели: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def compare_and_register_model(new_metrics, experiment_name):
    """
    Register model as challenger always.
    Only register as champion if no existing champion.
    """
    print(f"DEBUG: Registering model for experiment {experiment_name} as challenger")
    client = MlflowClient()
    model_name = f"{experiment_name}_model"
    # Check if model exists, create if not
    try:
        client.get_registered_model(model_name)
        print(f"Model '{model_name}' already registered")
    except Exception as e:
        print(f"DEBUG: Creating new registered model: {str(e)}")
        client.create_registered_model(model_name)
        print(f"Created new registered model '{model_name}'")

    model_versions = client.get_latest_versions(model_name)
    champion_exists = False
    for version in model_versions:
        if hasattr(version, 'aliases') and "champion" in version.aliases:
            champion_exists = True
            break
        elif hasattr(version, 'tags') and version.tags.get('alias') == "champion":
            champion_exists = True
            break

    run_id = new_metrics["run_id"]
    model_uri = f"runs:/{run_id}/model"
    print(f"DEBUG: Registering model from {model_uri}")
    model_details = mlflow.register_model(model_uri, model_name)
    new_version = model_details.version
    print(f"DEBUG: Registered new version: {new_version}")

    # Always set challenger alias
    try:
        if hasattr(client, 'set_registered_model_alias'):
            client.set_registered_model_alias(model_name, "challenger", new_version)
        else:
            client.set_model_version_tag(model_name, new_version, "alias", "challenger")
    except Exception as e:
        print(f"ERROR: Failed setting challenger alias: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # fallback
        client.set_model_version_tag(model_name, new_version, "alias", "challenger")

    print(f"Version {new_version} of model '{model_name}' set as 'challenger'")

    # If no champion, also set champion alias once
    if not champion_exists:
        try:
            if hasattr(client, 'set_registered_model_alias'):
                client.set_registered_model_alias(model_name, "champion", new_version)
            else:
                client.set_model_version_tag(model_name, new_version, "alias", "champion")
            print(f"Version {new_version} also set as 'champion' because no champion existed before")
        except Exception as e:
            print(f"ERROR: Failed setting champion alias: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")

    return True


def main():
    """Main function to run the fraud detection model training."""
    print("DEBUG: Скрипт запущен, начинаем инициализацию")
    parser = argparse.ArgumentParser(description="Fraud Detection Model Training")
    # Основные параметры
    parser.add_argument("--input", required=True, help="Input data path")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--model-type", default="rf", help="Model type (rf or lr)")

    # MLflow параметры
    parser.add_argument("--tracking-uri", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="fraud_detection", help="MLflow exp name")
    parser.add_argument("--auto-register", action="store_true", help="Automatically register")
    parser.add_argument("--run-name", default=None, help="Name for the MLflow run")

    # Отключение проверки Git для MLflow
    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

    # S3 параметры
    parser.add_argument("--s3-endpoint-url", help="S3 endpoint URL")
    parser.add_argument("--s3-access-key", help="S3 access key")
    parser.add_argument("--s3-secret-key", help="S3 secret key")

    args = parser.parse_args()
    print(f"DEBUG: Аргументы командной строки: {args}")

    # Настраиваем S3 конфигурацию
    s3_config = None
    if args.s3_endpoint_url and args.s3_access_key and args.s3_secret_key:
        print("DEBUG: Настраиваем S3 конфигурацию")
        s3_config = {
            'endpoint_url': args.s3_endpoint_url,
            'access_key': args.s3_access_key,
            'secret_key': args.s3_secret_key
        }
        os.environ['AWS_ACCESS_KEY_ID'] = args.s3_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = args.s3_secret_key
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_endpoint_url
        print("DEBUG: Переменные окружения для S3 установлены")

    if args.tracking_uri:
        print(f"DEBUG: Устанавливаем MLflow tracking URI: {args.tracking_uri}")
        mlflow.set_tracking_uri(args.tracking_uri)

    print(f"DEBUG: Устанавливаем MLflow эксперимент: {args.experiment_name}")
    mlflow.set_experiment(args.experiment_name)

    print("DEBUG: Создаем Spark сессию")
    spark = create_spark_session(s3_config).getOrCreate()
    print("DEBUG: Spark сессия создана")

    try:
        # Load and prepare data
        print("DEBUG: Загружаем данные")
        train_df, test_df = load_data(spark, args.input)

        print("DEBUG: Подготавливаем признаки")
        train_df, test_df, feature_cols = prepare_features(train_df, test_df)

        # Generate run name if not provided
        run_name = (
            args.run_name or f"fraud_detection_{args.model_type}_{os.path.basename(args.input)}"
        )
        print(f"DEBUG: Run name: {run_name}")

        # Train the model
        print("DEBUG: Обучаем модель")
        model, metrics = train_model(train_df, test_df, feature_cols, args.model_type, run_name)

        # Save the model locally
        print("DEBUG: Сохраняем модель")
        save_model(model, args.output)

        # Register model if enabled
        if args.auto_register:
            print("DEBUG: Сравниваем и регистрируем модель")
            compare_and_register_model(metrics, args.experiment_name)

        print("Training completed successfully!")

    except Exception as e:
        print(f"ERROR: Ошибка во время обучения: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        print("DEBUG: Останавливаем Spark сессию")
        spark.stop()
        print("DEBUG: Скрипт завершен")

if __name__ == "__main__":
    main()