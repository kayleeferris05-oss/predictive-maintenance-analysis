import pandas as pd

# Load the dataset
df = pd.read_csv("ai4i2020.csv")

# Preview the first few rows
print(df.head())

import pandas as pd

# Load the dataset
df = pd.read_csv("ai4i2020.csv")

# Preview the first few rows
print("--- First 5 Rows ---")
print(df.head())

# Check dataset structure, columns, and data types
print("\n--- Dataset Info ---")
print(df.info())

# Check for any missing/null values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Check the distribution of our target variable (Machine Failure)
print("\n--- Machine Failure Counts ---")
print(df["Machine failure"].value_counts())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("ai4i2020.csv")

# Set up the visual style
sns.set_theme(style="whitegrid")

# Create a scatter plot: Torque vs Tool Wear, colored by Machine Failure
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df, 
    x="Tool wear [min]", 
    y="Torque [Nm]", 
    hue="Machine failure", 
    palette={0: "skyblue", 1: "crimson"}, 
    alpha=0.8,
    s=40
)

# Add titles and labels
plt.title("Machine Failure Analysis: Torque vs. Tool Wear", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Tool Wear [min]", fontsize=12)
plt.ylabel("Torque [Nm]", fontsize=12)

# Customize legend
plt.legend(title="Status", labels=["Normal Operation (0)", "Machine Failure (1)"], loc="upper right")

# Save high-res image for your portfolio
plt.savefig("torque_vs_toolwear_analysis.png", dpi=300, bbox_inches='tight')
print("Portfolio chart saved successfully as 'torque_vs_toolwear_analysis.png'!")

# Display the plot window
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the dataset
df = pd.read_csv("ai4i2020.csv")

# 1. Feature Selection & Preparation
# Drop non-predictive ID columns and text columns for this base model
# 'Type' is categorical, so we can use one-hot encoding, or drop it for simplicity in the first baseline model
X = df.drop(columns=["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])
y = df["Machine failure"]

# 2. Train/Test Split (stratified to preserve the minority failure class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Initialize and Train the Random Forest Model
# class_weight='balanced' helps the model pay attention to the rare failure cases
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# 4. Predictions and Evaluation
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("--- Model Evaluation Results ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# 5. Feature Importance Analysis
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Feature Importances ---")
print(feature_importances)

