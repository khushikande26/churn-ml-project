from data_preprocessing import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

print("🚀 Starting training...")

# Load data
df = load_data("data/raw/churn.csv")
print("✅ Data loaded:", df.shape)

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)
print("✅ Data preprocessed")

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("✅ XGBoost Model trained")

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("🎯 Accuracy:", acc)

# Get feature importance
importance = model.feature_importances_
features = X_train.columns

# Plot
plt.figure(figsize=(10,5))
plt.barh(features, importance)
plt.title("Feature Importance (Churn Prediction)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Save model
joblib.dump(model, "models/model.pkl")
print("💾 Model saved!")