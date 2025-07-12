import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Sample data (Area in sqft, Bedrooms, Age in years, Price in ₹L)
data = pd.DataFrame({
    'area': [1000, 1500, 2000, 1200, 1800],
    'bedrooms': [2, 3, 4, 2, 3],
    'age': [5, 10, 15, 7, 12],
    'price': [50, 75, 100, 60, 85]
})

X = data[['area', 'bedrooms', 'age']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'real_estate_model.pkl')
print("Model saved as real_estate_model.pkl")

# Ensure the model is saved correctly
assert joblib.load('real_estate_model.pkl') is not None, "Model loading failed"
print("Model loaded successfully")      

# This code snippet is for training a simple linear regression model
# and saving it using joblib. It uses sample data for demonstration purposes.           
# This file is used to launch the IPython kernel for Jupyternotebooks.
# This file is used to launch the Flask application for the real estate prediction model.
# This file is used to launch the Flask application for the real estate prediction model.
# This file is used to launch the Flask application for the real estate prediction model.

