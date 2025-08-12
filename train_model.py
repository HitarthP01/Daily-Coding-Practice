import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Linear_Regression_Experiment")

def generate_data(n_samples=1000):
    """Gererate synthetic data for training."""
    np.random.seed(42)
    X = np.random.rand(n_samples,1) * 10
    y = 3 * X.flatten() + 2 + np.random.randn(n_samples) * 0.5
    return X,y

def train_model():
    """Train a linear regression model and log parameters and metrics."""
    X, y = generate_data()

    #start MLflow run
    with mlflow.start_run():
        #Train model
        model = LinearRegression()
        model.fit(X,y)

        #Predict and calculate metrics
        y_pred = model.predict(X)
        mse = mean_squared_error(y,y_pred)

        mlflow.log_param("n_samples",len(X))
        mlflow.log_metric("mse",mse)
        mlflow.sklearn.log_model(model,"linear_regression")

        print(f"Model trained with MSE: {mse:.4f}")
        return model
    
    # # Start MLflow run
    # with mlflow.start_run():
    #     # Initialize and train the model
    #     model = LinearRegression()
    #     model.fit(X, y)
        
    #     # Make predictions
    #     predictions = model.predict(X)
        
    #     # Calculate metrics
    #     mse = mean_squared_error(y, predictions)
        
    #     # Log parameters and metrics
    #     mlflow.log_param("model_type", "Linear Regression")
    #     mlflow.log_metric("mse", mse)
        
    #     # Log the model
    #     mlflow.sklearn.log_model(model, "model")
        
    #     print(f"Model trained with MSE: {mse}")

if __name__ == "__main__":
    train_model()
    print("Training complete. Check MLflow UI for details.")