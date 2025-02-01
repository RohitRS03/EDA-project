# Exploratory Data Analysis (EDA) Project
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Step 2: Load the Data
data_path = "healthcare_dataset.csv"
df = pd.read_csv(data_path)

# Step 3: Handle Missing and Duplicate Data
# Drop duplicates
df = df.drop_duplicates()

# Step 4: Data Overview
print("Dataset Information:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nUnique Values per Column:\n")
print(df.nunique())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\ntotal_billing_by_hospital\n")
print(df.describe())

print ("\navg_stay_duration_by_ medical_condition\n")
print(df.describe())

print("\npatients_by_medication_category\n")
print(df.describe())
# Fill missing values
for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=['number']):
    df[col].fillna(df[col].mean(), inplace=True)

# Step 5: Feature Engineering
# Convert date columns to datetime format
date_columns = ['Date of Admission', 'Discharge Date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=date_columns)

# Create new features
df['hospital_stay_duration'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['age_group'] = pd.cut(df['Age'], bins=[ 10, 40, 50, 60,100], labels=['Child',  'Adult', 'Middle Age', 'Senior'])
df['admission_year'] = df['Date of Admission'].dt.year
df['admission_month'] = df['Date of Admission'].dt.month
df['admission_weekday'] = df['Date of Admission'].dt.weekday
total_billing_by_hospital = df.groupby('Hospital')['Billing Amount'].sum()
department_categories = pd.qcut(total_billing_by_hospital, q=4, labels=['Low', 'Medium', 'High', 'Very High'])
df['hospital_category'] = df['Hospital'].map(department_categories)
avg_stay_duration_by_medical_condition = df.groupby('Medical Condition')['hospital_stay_duration'].mean().sort_values(
    ascending=False)
patients_by_medication_category = df.groupby('Medical Condition')['age_group'].nunique().sort_values(ascending=False)

# Step 6: Univariate Analysis
numerical_cols = ['Age', 'hospital_stay_duration', 'Billing Amount']
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col],color='r', bins=50, edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of {col}")
    plt.show()


categorical_cols = [ 'Blood Type', 'Test Results', 'age_group']
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    value_counts = df[col].value_counts()
    plt.bar(value_counts.index, value_counts.values, color='blue', edgecolor='black')
    plt.title(f"Distribution of {col}", style='italic')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


# Step 7: Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.barplot(x='age_group', y='Age', data=df, ci=None)
plt.title("Treatment by Age Group")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Insurance Provider', y='Billing Amount', data=df, ci=None, palette='viridis')
plt.title("Billing Amount by Insurance Provider")
plt.xticks(rotation=45)
plt.xlabel("Insurance Provider")
plt.ylabel("Billing Amount")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='admission_year', y='admission_month', data=df, ci=None,palette='magma')
plt.title("Admission by month")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(x='age_group', y='hospital_stay_duration', data=df, ci=None,palette='magma')
plt.title("Hospital Stay Duration by Gender")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='Age', y='Medical Condition', data=df, palette='cividis')
plt.title("Boxplot of Billing Amount by Treatment Effectiveness")
plt.show()

data = df[['Billing Amount', 'Age', 'hospital_stay_duration']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, cmap='coolwarm', edgecolor='black',linecolor="black", linewidths=0.7, fmt=".2f")
plt.title("Correlation Matrix for Healthcare Data")
plt.show()

# Step 10: Save Cleaned Data
df.to_csv("cleaned_healthcare_data.csv", index=False)
