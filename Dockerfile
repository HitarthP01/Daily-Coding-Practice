FROM python:3.9-slim

RUN pip install mlflow psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow_db/mlflow.db", "--default-artifact-root", "/mlruns"]
