import pandas as pd

# Dataset Load
df = pd.read_csv("train.csv")

# First 5 Rows
print("\n===== First 5 Rows =====")
print(df.head())

# Dataset Shape
print("\n===== Dataset Shape =====")
print(df.shape)

# Column Names
print("\n===== Columns =====")
print(df.columns)

# Dataset Information
print("\n===== Dataset Info =====")
print(df.info())

# Statistical Summary
print("\n===== Statistical Summary =====")
print(df.describe())

# Missing Values
print("\n===== Missing Values =====")
print(df.isnull().sum())
import pandas as pd

# Dataset Load
df = pd.read_csv("train.csv")

# Missing Values Before Cleaning
print("\n===== Missing Values Before Cleaning =====")
print(df.isnull().sum())

# Age column ki missing values median se fill
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked column ki missing values mode se fill
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin column drop (bahut zyada missing values hain)
df.drop("Cabin", axis=1, inplace=True)

# Missing Values After Cleaning
print("\n===== Missing Values After Cleaning =====")
print(df.isnull().sum())

# Cleaned Dataset Information
print("\n===== Cleaned Dataset Info =====")
print(df.info())

# First 5 Rows
print("\n===== First 5 Rows After Cleaning =====")
print(df.head())

# Save Cleaned Dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\n✅ Dataset Cleaned Successfully!")
print("✅ New File Created: cleaned_titanic.csv")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Cleaned Dataset
df = pd.read_csv("cleaned_titanic.csv")

# Graph Style
sns.set_style("whitegrid")

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# 3. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# 4. Age Distribution
plt.figure(figsize=(8,4))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# 5. Fare Distribution
plt.figure(figsize=(8,4))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

print("EDA Completed Successfully!")
import pandas as pd

# Load Cleaned Dataset
df = pd.read_csv("cleaned_titanic.csv")

# 1. Family Size Feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 2. IsAlone Feature
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# 3. Title Extraction from Name
df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

# Common Titles Keep Karo
df["Title"] = df["Title"].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major',
     'Rev','Sir','Jonkheer','Dona'],
    'Rare'
)

df["Title"] = df["Title"].replace('Mlle', 'Miss')
df["Title"] = df["Title"].replace('Ms', 'Miss')
df["Title"] = df["Title"].replace('Mme', 'Mrs')

# Check New Features
print(df[["Name","FamilySize","IsAlone","Title"]].head())

# Save Dataset
df.to_csv("featured_titanic.csv", index=False)

print("\n✅ Feature Engineering Completed!")
print("✅ New File Created: featured_titanic.csv")
import pandas as pd

# Load Featured Dataset
df = pd.read_csv("featured_titanic.csv")

# Sex Encoding
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# Embarked Encoding
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

# Title Encoding
df["Title"] = df["Title"].map({
    "Mr": 0,
    "Miss": 1,
    "Mrs": 2,
    "Master": 3,
    "Rare": 4
})

# Check Data
print(df.head())

# Save Encoded Dataset
df.to_csv("encoded_titanic.csv", index=False)

print("\n✅ Encoding Completed Successfully!")
print("✅ New File Created: encoded_titanic.csv")
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Encoded Dataset
df = pd.read_csv("encoded_titanic.csv")

# Features (Input)
X = df[[
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Embarked",
    "Title"
]]

# Target (Output)
y = df["Survived"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Check Shapes
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

print("\n✅ Train-Test Split Completed!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("encoded_titanic.csv")

# Features
X = df[[
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Embarked",
    "Title"
]]

# Target
y = df["Survived"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestClassifier(random_state=42)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n✅ Model Training Completed!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load Dataset
df = pd.read_csv("encoded_titanic.csv")

# Features
X = df[[
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Embarked",
    "Title"
]]

# Target
y = df["Survived"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n✅ Model Evaluation Completed!")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("encoded_titanic.csv")

# Features
X = df[[
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Embarked",
    "Title"
]]

# Target
y = df["Survived"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== Feature Importance =====")
print(importance)

# Graph
plt.figure(figsize=(8,5))
plt.bar(importance["Feature"], importance["Importance"])
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

print("\n✅ Feature Importance Analysis Completed!")
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("encoded_titanic.csv")

# Features
X = df[[
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Embarked",
    "Title"
]]

# Target
y = df["Survived"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save Model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model Saved Successfully!")
print("✅ File Created: model.pkl")