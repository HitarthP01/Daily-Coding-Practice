## Technical Details

### Airflow Configuration
- **Version**: Uses Apache Airflow 2.10.2, specified in the `Dockerfile` (`FROM apache/airflow:2.10.2`) and `requirements.txt` (`apache-airflow==2.10.2`).
- **Executor**: Configured with `RedisExecutor` (set via `AIRFLOW__CORE__EXECUTOR`) for distributed task execution, replacing the default `SequentialExecutor`. This allows multiple tasks to run concurrently using Redis as a message broker.
- **Database**:
  - Metadata is stored in SQLite (`airflow.db`) with `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db`.
  - The `load` task in the DAG creates a custom `results` table using SQLite commands:
    ```sql
    CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, avg_price FLOAT);
    INSERT INTO results (avg_price) VALUES (?);
Docker Compose Configuration Overview
Your docker-compose.yml uses version 3.8 syntax. Note that the version attribute is deprecated but safe to ignore for now.

Services
airflow-webserver

Exposes port 8080:8080 for Airflow UI access.

Mounts local ./dags and ./data folders as volumes to persist DAG code and data.

Depends on airflow-scheduler and airflow-init services to ensure proper startup order.

airflow-scheduler

Handles task scheduling and execution in Airflow.

Depends on airflow-init for DB initialization.

airflow-init

Runs database migrations with airflow db migrate.

Creates an admin user for Airflow UI access.

Runs a loop (while sleep 3600; do :; done) to stay alive for health checks.

redis

Uses official redis:7.0 image.

Exposes port 6379:6379.

Mounts ./redis local folder to /data in the container to persist Redis data.

Includes a health check using redis-cli ping with configured intervals and retries.

Health Checks
airflow-init: Runs

bash
Copy
Edit
sqlite3 /opt/airflow/airflow.db .tables
to verify DB initialization.

redis: Runs

bash
Copy
Edit
redis-cli ping
to confirm Redis responsiveness.

Health checks are configured with:

Interval: 5 seconds

Timeout: 3 seconds

Retries: 5

Start period: 10 seconds (allows initial startup grace)

Volumes
./dags:/opt/airflow/dags — persist your DAG Python files

./data:/opt/airflow/data — persist your data files (e.g., CSVs)

./redis:/data — persist Redis database files

Airflow DAG Implementation (data_pipeline.py)
DAG Structure and Parameters
python
Copy
Edit
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import sqlite3

with DAG(
    dag_id="data_pipeline",
    start_date=datetime(2025, 8, 7),
    schedule_interval=None,
    catchup=False,
) as dag:
dag_id: Unique identifier for the pipeline.

start_date: When the DAG starts (set to a past date to allow manual triggering).

schedule_interval=None: Disables automatic scheduling — triggers must be manual.

catchup=False: Prevents Airflow from running missed DAG runs when turned on after inactivity.

Tasks
1. Extract
python
Copy
Edit
def extract():
    df = pd.read_csv("/opt/airflow/data/sample_data.csv")
    return df.to_dict()
Reads a CSV file into a Pandas DataFrame.

Returns data as a dictionary for passing via XCom.

2. Transform
python
Copy
Edit
def transform(**context):
    df = pd.DataFrame(context["task_instance"].xcom_pull(task_ids="extract"))
    avg_price = df["price"].mean()
    return {"avg_price": avg_price}
Pulls extracted data using XCom.

Calculates average of the price column.

Returns the result as a dictionary.

3. Load
python
Copy
Edit
def load(**context):
    avg_price = context["task_instance"].xcom_pull(task_ids="transform")["avg_price"]
    conn = sqlite3.connect("/opt/airflow/airflow.db")
    conn.execute("CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, avg_price FLOAT)")
    conn.execute("INSERT INTO results (avg_price) VALUES (?)", (avg_price,))
    conn.commit()
    conn.close()
Pulls transformed average price from XCom.

Inserts it into an SQLite table named results.

Task Dependencies
python
Copy
Edit
extract_task >> transform_task >> load_task
Defines execution order: extract → transform → load.

Additional Notes and Tips
Understanding Git Pull/Rebase vs Merge
Merge creates a new commit combining histories, which can clutter logs.

Rebase rewrites your commits on top of upstream changes, keeping history linear.

Use rebase for cleaner history, especially on feature branches.

Docker Compose Tips
Use named volumes for better data persistence in production.

Health checks are critical to ensure services are ready before dependent services start.

For complex setups, consider adding environment variables and secrets management.

Airflow Tips
Keep DAGs idempotent and stateless for reliability.

Use Airflow Variables and Connections to store credentials securely.

Monitor Airflow logs for debugging task failures.
