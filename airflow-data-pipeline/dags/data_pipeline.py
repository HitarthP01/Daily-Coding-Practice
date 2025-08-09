# from airflow import DAG  # DAG is the main Airflow object for defining workflows
# from airflow.operators.python import PythonOperator  # Operator to run Python functions as tasks
# from datetime import datetime  # Used to set the DAG's start date
# import pandas as pd  # For data manipulation
# import sqlite3  # For interacting with SQLite databases

# # Task 1: Extract data from a CSV file
# def extract_data():
#     # Read data from CSV file using pandas
#     df = pd.read_csv('/opt/airflow/data/sample_data.csv')
#     # Convert DataFrame to dictionary for XCom transfer between tasks
#     return df.to_dict()

# # Task 2: Transform data (calculate average price)
# def transform_data(**context):
#     # Retrieve data from previous task using XCom (Airflow's inter-task communication)
#     df = pd.DataFrame(context['task_instance'].xcom_pull(task_ids='extract'))
#     # Calculate the mean of the 'price' column
#     avg_prices = df["price"].mean()
#     # Return the result as a dictionary for XCom
#     return {'average_price': avg_prices}

# # Task 3: Load data into SQLite database
# def load_data(**context):
#     # Pull the average price from the previous task using XCom
#     avg_price = context['task_instance'].xcom_pull(task_ids='transform_data')["avg_price"]
#     # Connect to SQLite database (creates file if it doesn't exist)
#     conn = sqlite3.connect('/opt/airflow/airflow.db')
#     # Create table if it doesn't exist
#     conn.execute("CREATE TABLE IF NOT EXISTS results(id INTEGER PRIMARY KEY, avg_price FLOAT)")
#     # Insert the average price into the table
#     conn.execute("INSERT INTO results (avg_price) VALUES (?)", (avg_price,))
#     # Commit changes and close connection
#     conn.commit()
#     conn.close()

# # Define the DAG (Directed Acyclic Graph) - the workflow itself
# with DAG(
#     dag_id='data_pipeline',  # Unique identifier for the DAG
#     start_date=datetime(2025, 8, 7),  # When to start running the DAG
#     schedule_interval=None,  # Run only when triggered manually
#     catchup=False,  # Do not run missed intervals since start_date
# ) as dag:
#     # Define the extract task using PythonOperator
#     extract = PythonOperator(
#         task_id='extract_data',  # Unique task name
#         python_callable=extract_data,  # Function to run
#     )

#     # Define the transform task
#     transform = PythonOperator(
#         task_id='transform_data',
#         python_callable=transform_data,
#         provide_context=True,  # Pass Airflow context to the function
#     )

#     # Define the load task
#     load = PythonOperator(
#         task_id='load_data',
#         python_callable=load_data,
#         provide_context=True,
#     )

#     # Set task dependencies (order of execution)
#     extract >> transform >> load  # extract_data runs first, then transform_data, then load_data

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import sqlite3

def extract():
    """Read data from CSV."""
    df = pd.read_csv("/opt/airflow/data/sample_data.csv")
    return df.to_dict()

def transform(**context):
    """Calculate average price."""
    df = pd.DataFrame(context["task_instance"].xcom_pull(task_ids="extract"))
    avg_price = df["price"].mean()
    return {"avg_price": avg_price}

def load(**context):
    """Load results into SQLite."""
    avg_price = context["task_instance"].xcom_pull(task_ids="transform")["avg_price"]
    conn = sqlite3.connect("/opt/airflow/airflow.db")
    conn.execute("CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, avg_price FLOAT)")
    conn.execute("INSERT INTO results (avg_price) VALUES (?)", (avg_price,))
    conn.commit()
    conn.close()

# Define the DAG
with DAG(
    dag_id="data_pipeline",
    start_date=datetime(2025, 8, 7),
    schedule_interval=None,
    catchup=False,
) as dag:
    extract_task = PythonOperator(task_id="extract", python_callable=extract)
    transform_task = PythonOperator(task_id="transform", python_callable=transform)
    load_task = PythonOperator(task_id="load", python_callable=load)

    extract_task >> transform_task >> load_task