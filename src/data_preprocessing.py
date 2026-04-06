import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path, encoding='latin1')
    return df

def preprocess_data(df):
    # Drop customerID
    df = df.drop("customerID", axis=1)

    # Fix TotalCharges (very common issue)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Convert target column
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical features
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        if col != "Churn":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Split features & target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.2, random_state=42)