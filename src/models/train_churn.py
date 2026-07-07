import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Load & clean ---
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

cols_to_simplify = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                     'StreamingTV', 'StreamingMovies', 'MultipleLines']
for col in cols_to_simplify:
    df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

df_clean = df.drop(columns=['customerID'])
df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)
df_clean.to_csv('data/processed/telco_churn_cleaned.csv', index=False)

# --- Features/target ---
X = df_clean.drop(columns=['Churn'])
y = df_clean['Churn']
numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [c for c in X.columns if c not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_features),
])

params = {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced", "random_state": 42}
pipe = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(**params))])

# --- Train & log ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("customer-churn")

with mlflow.start_run(run_name="random_forest_baseline"):
    print("Training...")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))

    print("Logging model...")
    mlflow.sklearn.log_model(pipe, "model", registered_model_name="churn-model")

    print("Done. View at http://localhost:5000")
