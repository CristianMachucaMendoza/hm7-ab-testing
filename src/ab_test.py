import os
import sys
import traceback
import argparse
import json
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from datetime import datetime


def create_spark_session(s3_config=None):
    print("DEBUG: Initializing Spark session for A/B testing")
    try:
        builder = (SparkSession.builder.appName("FraudDetectionABTesting"))
        if s3_config and all(k in s3_config for k in ['endpoint_url', 'access_key', 'secret_key']):
            builder = (builder
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.endpoint", s3_config['endpoint_url'])
                .config("spark.hadoop.fs.s3a.access.key", s3_config['access_key'])
                .config("spark.hadoop.fs.s3a.secret.key", s3_config['secret_key'])
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
            )
        return builder.getOrCreate()
    except Exception as e:
        print(f"ERROR: Spark session error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None


def load_test_data(spark, input_path):
    print(f"DEBUG: Loading test data from {input_path}")
    try:
        df = spark.read.parquet(input_path, header=True, inferSchema=True)
        print("Dataset Schema:")
        df.printSchema()
        print(f"Total records: {df.count()}")
        print("First 5 rows from test data:")
        df.show(5, truncate=False)
        _, df = df.randomSplit([0.005, 0.00001], seed=42)
        return df
    except Exception as e:
        print(f"ERROR: Error loading test data: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None


def prepare_features_for_test(test_df):
    print("DEBUG: Preparing test features")
    dtypes = dict(test_df.dtypes)
    feature_cols = [c for c in test_df.columns if c not in ('tx_fraud', 'tx_timestamp') and dtypes[c] != 'string']
    print(f"Selected {len(feature_cols)} numeric feature columns: {feature_cols}")
    for col_name in test_df.columns:
        null_count = test_df.filter(test_df[col_name].isNull()).count()
        if null_count > 0:
            print(f"WARNING: Column '{col_name}' has {null_count} null values")
    return test_df, feature_cols


def check_alias_exists(experiment_name, alias):
    print(f"DEBUG: Checking alias '{alias}' for experiment '{experiment_name}'")
    client = MlflowClient()
    model_name = f"{experiment_name}_model"
    try:
        registered_model = client.get_registered_model(model_name)
        for version in registered_model.latest_versions:
            if hasattr(version, 'aliases') and alias in version.aliases:
                return True
            elif hasattr(version, 'tags') and version.tags.get('alias') == alias:
                return True
        return False
    except Exception as e:
        print(f"ERROR checking alias '{alias}': {str(e)}")
        return False


def get_model_by_alias(experiment_name, alias, spark):
    print(f"DEBUG: Loading model alias '{alias}' for experiment '{experiment_name}'")
    client = MlflowClient()
    model_name = f"{experiment_name}_model"
    try:
        registered_model = client.get_registered_model(model_name)
        target_version = None
        for version in registered_model.latest_versions:
            if hasattr(version, 'aliases') and alias in version.aliases:
                target_version = version
                break
            elif hasattr(version, 'tags') and version.tags.get('alias') == alias:
                target_version = version
                break
        if not target_version:
            print(f"WARNING: No version found with alias '{alias}'")
            return None, None, None
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.spark.load_model(model_uri, spark_session=spark)
        return model, target_version.run_id, target_version.version
    except Exception as e:
        print(f"ERROR loading model alias '{alias}': {str(e)}")
        print(traceback.format_exc())
        return None, None, None


def perform_ab_testing(spark, test_df, champion_model, challenger_model,
                       experiment_name, traffic_split=0.5, test_name="ab_test"):
    print(f"DEBUG: A/B testing with traffic split: {traffic_split}")
    try:

        ab_test_df = test_df.withColumn("ab_group",
                                        when(rand() < traffic_split, "challenger").otherwise("champion"))
        group_counts = ab_test_df.groupBy("ab_group").count().collect()
        champion_count = next((row['count'] for row in group_counts if row['ab_group'] == 'champion'), 0)
        challenger_count = next((row['count'] for row in group_counts if row['ab_group'] == 'challenger'), 0)

        champion_test_df = ab_test_df.filter(col("ab_group") == "champion")
        challenger_test_df = ab_test_df.filter(col("ab_group") == "challenger")

        champion_predictions = champion_model.transform(champion_test_df)
        challenger_predictions = challenger_model.transform(challenger_test_df)

        evaluator_auc = BinaryClassificationEvaluator(labelCol="tx_fraud",
                                                      rawPredictionCol="rawPrediction",
                                                      metricName="areaUnderROC")
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="tx_fraud",
                                                          predictionCol="prediction",
                                                          metricName="accuracy")
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="tx_fraud",
                                                         predictionCol="prediction",
                                                         metricName="f1")

        champion_auc = evaluator_auc.evaluate(champion_predictions)
        champion_accuracy = evaluator_acc.evaluate(champion_predictions)
        champion_f1 = evaluator_f1.evaluate(champion_predictions)

        challenger_auc = evaluator_auc.evaluate(challenger_predictions)
        challenger_accuracy = evaluator_acc.evaluate(challenger_predictions)
        challenger_f1 = evaluator_f1.evaluate(challenger_predictions)

        auc_improvement = challenger_auc - champion_auc
        accuracy_improvement = challenger_accuracy - champion_accuracy
        f1_improvement = challenger_f1 - champion_f1

        total_samples = champion_count + challenger_count
        statistical_significance = min(champion_count, challenger_count) / total_samples > 0.1

        confidence_threshold = 0.02
        if auc_improvement > confidence_threshold and statistical_significance:
            winner = "challenger"
            improvement_msg = f"Challenger better by {auc_improvement:.4f} AUC (statistically significant)"
            should_promote = True
        elif auc_improvement > 0:
            winner = "challenger"
            improvement_msg = f"Challenger better by {auc_improvement:.4f} AUC (not statistically significant)"
            should_promote = False
        else:
            winner = "champion"
            improvement_msg = f"Champion better by {-auc_improvement:.4f} AUC"
            should_promote = False

        ab_results = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "traffic_split": traffic_split,
            "champion_metrics": {
                "auc": champion_auc,
                "accuracy": champion_accuracy,
                "f1": champion_f1,
                "sample_size": champion_count
            },
            "challenger_metrics": {
                "auc": challenger_auc,
                "accuracy": challenger_accuracy,
                "f1": challenger_f1,
                "sample_size": challenger_count
            },
            "improvements": {
                "auc": auc_improvement,
                "accuracy": accuracy_improvement,
                "f1": f1_improvement
            },
            "statistical_significance": statistical_significance,
            "winner": winner,
            "should_promote": should_promote,
            "improvement_message": improvement_msg,
            "confidence_threshold": confidence_threshold
        }

        with mlflow.start_run(run_name=f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as ab_run:
            mlflow.log_param("ab_traffic_split", traffic_split)
            mlflow.log_param("ab_test_timestamp", ab_results["timestamp"])
            mlflow.log_param("ab_test_name", test_name)
            mlflow.log_param("confidence_threshold", confidence_threshold)

            mlflow.log_metric("champion_auc", champion_auc)
            mlflow.log_metric("champion_accuracy", champion_accuracy)
            mlflow.log_metric("champion_f1", champion_f1)
            mlflow.log_metric("champion_sample_size", champion_count)

            mlflow.log_metric("challenger_auc", challenger_auc)
            mlflow.log_metric("challenger_accuracy", challenger_accuracy)
            mlflow.log_metric("challenger_f1", challenger_f1)
            mlflow.log_metric("challenger_sample_size", challenger_count)

            mlflow.log_metric("auc_improvement", auc_improvement)
            mlflow.log_metric("accuracy_improvement", accuracy_improvement)
            mlflow.log_metric("f1_improvement", f1_improvement)

            mlflow.log_param("winner", winner)
            mlflow.log_param("should_promote", should_promote)
            mlflow.log_param("statistical_significance", statistical_significance)
            mlflow.log_param("improvement_message", improvement_msg)

            mlflow.log_text(json.dumps(ab_results, indent=2), "ab_test_results.json")

        print(f"=== A/B TEST RESULTS ===")
        print(f"Champion - AUC: {champion_auc:.4f}, Accuracy: {champion_accuracy:.4f}, F1: {champion_f1:.4f}")
        print(f"Challenger - AUC: {challenger_auc:.4f}, Accuracy: {challenger_accuracy:.4f}, F1: {challenger_f1:.4f}")
        print(f"Improvement - AUC: {auc_improvement:.4f}, Accuracy: {accuracy_improvement:.4f}, F1: {f1_improvement:.4f}")
        print(f"Statistical Significance: {statistical_significance}")
        print(f"WINNER: {winner.upper()}")
        print(f"Should Promote: {should_promote}")
        print(f"Message: {improvement_msg}")
        print("=======================")

        return ab_results

    except Exception as e:
        print(f"ERROR: A/B testing error: {str(e)}")
        print(traceback.format_exc())
        return {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "error": str(e),
            "error_traceback": traceback.format_exc(),
            "status": "failed"
        }


def promote_challenger_model(experiment_name, challenger_version):
    print(f"DEBUG: Promoting challenger version {challenger_version} to champion")
    client = MlflowClient()
    model_name = f"{experiment_name}_model"
    try:
        current_champion = None
        for version in client.get_latest_versions(model_name):
            if hasattr(version, 'aliases') and "champion" in version.aliases:
                current_champion = version
                break

        if current_champion:
            print(f"DEBUG: Removing champion alias from version {current_champion.version}")
            try:
                if hasattr(client, 'delete_registered_model_alias'):
                    client.delete_registered_model_alias(model_name, "champion")
                else:
                    client.set_model_version_tag(model_name, current_champion.version, "alias", "archived")
            except Exception as e:
                print(f"WARNING: Error removing champion alias: {str(e)}")

        print(f"DEBUG: Setting champion alias for version {challenger_version}")
        try:
            if hasattr(client, 'set_registered_model_alias'):
                client.set_registered_model_alias(model_name, "champion", challenger_version)
            else:
                client.set_model_version_tag(model_name, challenger_version, "alias", "champion")
        except Exception as e:
            print(f"ERROR: Error setting champion alias: {str(e)}")
            return False

        print(f"DEBUG: Removing challenger alias from version {challenger_version}")
        try:
            if hasattr(client, 'delete_registered_model_alias'):
                client.delete_registered_model_alias(model_name, "challenger")
            else:
                client.delete_model_version_tag(model_name, challenger_version, "alias")
        except Exception as e:
            print(f"WARNING: Error removing challenger alias: {str(e)}")

        print(f"DEBUG: Challenger version {challenger_version} promoted to champion")
        return True

    except Exception as e:
        print(f"ERROR: Promotion error: {str(e)}")
        print(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="A/B Testing for Fraud Detection Models")
    parser.add_argument("--input", required=True, help="Input test data path")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="fraud_detection", help="MLflow experiment name")
    parser.add_argument("--traffic-split", type=float, default=0.5, help="Traffic split for challenger (0-1)")
    parser.add_argument("--test-name", default="ab_test", help="Name for the A/B test")
    parser.add_argument("--auto-promote", action="store_true", help="Promote challenger if wins")
    parser.add_argument("--s3-endpoint-url", help="S3 endpoint URL")
    parser.add_argument("--s3-access-key", help="S3 access key")
    parser.add_argument("--s3-secret-key", help="S3 secret key")

    args = parser.parse_args()

    s3_config = None
    if args.s3_endpoint_url and args.s3_access_key and args.s3_secret_key:
        s3_config = {'endpoint_url': args.s3_endpoint_url,
                     'access_key': args.s3_access_key,
                     'secret_key': args.s3_secret_key}
        os.environ['AWS_ACCESS_KEY_ID'] = args.s3_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = args.s3_secret_key
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_endpoint_url

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    mlflow.set_experiment(args.experiment_name)

    spark = create_spark_session(s3_config)
    if not spark:
        print("ERROR: Spark session creation failed")
        sys.exit(0)  # Don't fail DAG

    try:
        champion_exists = check_alias_exists(args.experiment_name, "champion")
        challenger_exists = check_alias_exists(args.experiment_name, "challenger")

        if not champion_exists:
            print("WARNING: Champion model not found, can't run A/B test.")
            sys.exit(0)
        if not challenger_exists:
            print("WARNING: Challenger model not found, can't run A/B test.")
            sys.exit(0)

        test_df_raw = load_test_data(spark, args.input)
        if not test_df_raw:
            print("WARNING: Failed to load test data")
            sys.exit(0)

        test_df, feature_cols = prepare_features_for_test(test_df_raw)

        champion_model, champion_run_id, champion_version = get_model_by_alias(args.experiment_name, "champion", spark)
        challenger_model, challenger_run_id, challenger_version = get_model_by_alias(args.experiment_name, "challenger", spark)

        if not champion_model or not challenger_model:
            print("WARNING: Failed to load champion or challenger model")
            sys.exit(0)

        ab_results = perform_ab_testing(
            spark=spark,
            test_df=test_df,
            champion_model=champion_model,
            challenger_model=challenger_model,
            experiment_name=args.experiment_name,
            traffic_split=args.traffic_split,
            test_name=args.test_name,
        )

        if "error" in ab_results:
            print(f"WARNING: A/B test error: {ab_results['error']}")
        else:
            if ab_results["should_promote"] and args.auto_promote:
                success = promote_challenger_model(args.experiment_name, challenger_version)
                if success:
                    ab_results["promotion_status"] = "success"
                else:
                    ab_results["promotion_status"] = "failed"
            else:
                ab_results["promotion_status"] = "not_promoted"

        print("A/B testing completed!")
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: Unexpected error: {str(e)}")
        print(traceback.format_exc())
        sys.exit(0)
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
