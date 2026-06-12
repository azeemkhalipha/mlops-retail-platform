import pickle
import os
from sklearn.linear_model import LinearRegression
import numpy as np

os.makedirs("models", exist_ok=True)
X = np.random.rand(1000, 7)
y = np.random.rand(1000) * 50
model = LinearRegression().fit(X, y)

with open("models/demand_forecast_v1.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model created successfully")
