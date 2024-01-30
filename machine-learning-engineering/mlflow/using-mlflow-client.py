from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor


client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
