import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_titanic.csv")

sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

sns.barplot(x="Embarked", y="Survived", data=df)
plt.title("Embarked vs Survival")
plt.show()

sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age vs Survival")
plt.show()

sns.boxplot(x="Survived", y="Fare", data=df)
plt.title("Fare vs Survival")
plt.show()

df = pd.read_csv("featured_titanic.csv")

sns.barplot(x="FamilySize", y="Survived", data=df)
plt.title("Family Size vs Survival")
plt.show()

import numpy as np

df = pd.read_csv("encoded_titanic.csv")

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()