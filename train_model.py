import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load Kaggle dataset
df = pd.read_csv("train.csv")

# Select features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'Neighborhood']
target = 'SalePrice'

X = df[features].copy
y = df[target]

# Handle missing values
X['BedroomAbvGr'].fillna(X['BedroomAbvGr'].median(), inplace=True)
X['FullBath'].fillna(X['FullBath'].median(), inplace=True)
X['Neighborhood'].fillna(X['Neighborhood'].mode()[0], inplace=True)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Neighborhood'])
    ],
    remainder='passthrough'
)

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
