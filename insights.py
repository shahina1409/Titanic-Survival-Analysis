import pandas as pd

df = pd.read_csv("cleaned_titanic.csv")

print("===== BUSINESS INSIGHTS =====")

# Overall Survival Rate
survival_rate = df["Survived"].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.2f}%")

# Female Survival
female_rate = df[df["Sex"]=="female"]["Survived"].mean()*100
print(f"Female Survival Rate: {female_rate:.2f}%")

# Male Survival
male_rate = df[df["Sex"]=="male"]["Survived"].mean()*100
print(f"Male Survival Rate: {male_rate:.2f}%")

# Best Passenger Class
class_rate = df.groupby("Pclass")["Survived"].mean()*100
print("\nSurvival Rate by Class:")
print(class_rate)

# Average Fare
print("\nAverage Fare:", round(df["Fare"].mean(),2))