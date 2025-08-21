import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

#! Be aware it uses 100% cpu

#? Load CSV
df = pd.read_csv("./data/medicines.csv")

#? Drop irrelevant / ID-like fields
X = df.drop(["SL", "Unit Price", "DAR No", "Trade Name"], axis=1)
y = df["Unit Price"]

#? Log-transform target to handle large price variance
y = np.log1p(y)

#? Categorical columns
cat_cols = ["Company", "Generice Name With Strength", "Dosage From", "Pack Size"]

#? Preprocessing: One-hot encode categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

#? Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

#? Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#? Train model
model.fit(X_train, y_train)

#? Predict on test
y_pred = model.predict(X_test)

#? Evaluate
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))  #? back-transform to Taka

print(f"Accuracy (R²): {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f} Taka")

#? Save trained model
joblib.dump(model, "medicine_price_model.pkl")
print("✅ Model saved as medicine_price_model.pkl")

# ------------------------------
#? Example: Predict new medicine
# ------------------------------
example = pd.DataFrame([{
    "Company": "Advanced Chemical Industries Limited",
    "Generice Name With Strength": "Paracetamol 500 mg",
    "Dosage From": "Tablet",
    "Pack Size": "10 x 10 in Blister Pack"
}])

predicted_log_price = model.predict(example)[0]
predicted_price = np.expm1(predicted_log_price)  # back-transform

print("\n💊 Predicted price for example medicine:")
print(predicted_price, "Taka")

