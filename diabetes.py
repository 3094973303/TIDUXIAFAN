import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Clinical Integration Guidelines
def print_clinical_integration_guidelines():
    print("\n" + "=" * 50)
    print("CLINICAL INTEGRATION GUIDELINES")
    print("=" * 50)
    print("""
    Based on this project's findings, we recommend integrating this machine learning tool 
    into clinical workflows as follows:

    1. Screening Phase:
       - Collect patient data during routine checkups
       - Use this system for preliminary risk assessment
       - Schedule further diabetes screening tests for high-risk patients

    2. Risk Stratification:
       - High Risk (>70%): Immediate referral for professional evaluation
       - Moderate Risk (40-70%): Intervention and follow-up within 6 months
       - Low Risk (<40%): Preventive advice and annual assessment

    3. Clinical Decision Support:
       - Use prediction results as an aid to clinical decisions
       - Physicians should combine results with patient history and other clinical findings
       - This tool cannot replace professional medical judgment

    4. Data Integration:
       - Record risk assessment results in electronic health record systems
       - Establish tracking mechanisms to monitor high-risk patients

    5. Continuous Improvement:
       - Update the model regularly with new data
       - Collect clinical feedback to optimize the algorithm
       - Evaluate the clinical effectiveness of the tool annually
    """)
    print("=" * 50)


# 1. Data loading and exploration
df = pd.read_csv(r"D:\AAterm2\524 ML\Group\1\Dataset of Diabetes .csv")

# View basic data information
print("Data dimensions:", df.shape)
print("\nFirst 5 rows of data:")
print(df.head())
print("\nStatistical information on data:")
print(df.describe())
print("\nMissing value statistics:")
print(df.isnull().sum())


# 2. Data preprocessing
# Handle gender encoding
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
# Handle class variable encoding (combining pre-diabetic and diabetic categories)
if 'CLASS' in df.columns:
    if df['CLASS'].dtype == object:
        df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 1})


# Handle missing values using median imputation
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Feature selection
features = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender']
target = 'CLASS'

# Ensure all features exist in the dataset
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"Warning: The following features are missing from the dataset: {missing_features}")
    print("Please check your dataset or modify the feature list. Available columns: ", df.columns.tolist())
    features = [f for f in features if f in df.columns]
    print(f"Will continue with these features: {features}")

# Standardize continuous features
scaler = StandardScaler()
X = scaler.fit_transform(df_imputed[features])
y = df_imputed[target]

# Split into training and test sets (80-20 split with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining model with the following features: {features}")
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")


# 3. Model training and comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

# Train each model and evaluate performance
results = {}
for name, model in models.items():
    print(f"\nTraining model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_pred)
    }


# 4. Results visualization and evaluation
# Output performance comparison
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df)

# Find the best performing model based on F1 score
best_model_name = results_df['F1'].idxmax()
print(f"\nBest model based on F1 score: {best_model_name}")

# Create class distribution visualization
plt.figure(figsize=(10, 6))
class_counts = df_imputed[target].value_counts()
colors = ['green', 'red'] if len(class_counts) == 2 else ['green', 'orange', 'red']
ax = class_counts.plot(kind='bar', color=colors)
plt.title('Distribution of Diabetes Cases', fontsize=16)
plt.xlabel('Class (0: Non-diabetic, 1: Pre-diabetic & Diabetic)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

# Add value labels
for i, v in enumerate(class_counts):
    ax.text(i, v + 5, str(v), ha='center', fontsize=12)

# Add percentage labels
total = class_counts.sum()
for i, v in enumerate(class_counts):
    percentage = v / total * 100
    ax.text(i, v/2, f"{percentage:.1f}%", ha='center', color='white', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
plt.close()
print("\nClass distribution plot saved as: class_distribution.png")

# Create feature importance visualization (using Random Forest)
rf = models['Random Forest']
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance - Random Forest", fontsize=16)
plt.bar(range(len(features)), importances[indices], align='center', color='dodgerblue')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Importance", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add value labels
for i, v in enumerate(importances[indices]):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
plt.close()
print("\nFeature importance plot saved as: feature_importance.png")

# Create confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks([0.5, 1.5], ['Non-diabetic (0)', 'Diabetic (1)'], fontsize=12)
plt.yticks([0.5, 1.5], ['Non-diabetic (0)', 'Diabetic (1)'], fontsize=12, rotation=0)

# Add percentage labels to confusion matrix
total = cm.sum()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        percentage = cm[i, j] / total * 100
        plt.text(j + 0.5, i + 0.7, f"{percentage:.1f}%", ha='center',
                 color='black' if cm[i, j] < cm.max() / 2 else 'white', fontsize=10)

plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
plt.close()
print("\nConfusion matrix saved as: confusion_matrix.png")

# Generate correlation analysis heatmap
print("\nGenerating correlation heatmap...")
# Add target variable to correlation analysis
corr_features = features.copy()
if target in df.columns:
    corr_features.append(target)

# Calculate correlation matrix
corr_matrix = df_imputed[corr_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("Correlation heatmap saved as: correlation_heatmap.png")


# 5. Best model selection and saving
best_model = models[best_model_name]

# Generate classification report
y_pred = best_model.predict(X_test)
print("\nBest Model Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and related files
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(features, 'models/features.pkl')

print(f"\nModel and related files saved to 'models' directory. Best model: {best_model_name}")

# Output model file size information
model_file = 'models/diabetes_model.pkl'
print(f"Model file size: {os.path.getsize(model_file) / (1024*1024):.2f} MB")

# Output clinical integration guidelines
print_clinical_integration_guidelines()

# Output web interface launch instructions
print("\n" + "=" * 50)
print("LAUNCHING THE WEB INTERFACE")
print("=" * 50)
print("""
To launch the diabetes risk assessment web interface, run the following command:

    streamlit run app.py

Make sure you have the Streamlit library installed:

    pip install streamlit

The web interface provides a user-friendly risk assessment tool suitable for healthcare professionals.
""")
print("=" * 50)