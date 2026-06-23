import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

# Load Dataset
df = pd.read_csv("cleaned_titanic.csv")

print("===== Statistical Analysis =====")

# --------------------------------
# 1. Gender vs Survival
# --------------------------------

gender_table = pd.crosstab(df["Sex"], df["Survived"])

chi2, p, dof, expected = chi2_contingency(gender_table)

print("\nGender vs Survival")
print("P-value:", p)

if p < 0.05:
    print("Significant Relationship Found")
else:
    print("No Significant Relationship")

# --------------------------------
# 2. Passenger Class vs Survival
# --------------------------------

class_table = pd.crosstab(df["Pclass"], df["Survived"])

chi2, p, dof, expected = chi2_contingency(class_table)

print("\nPassenger Class vs Survival")
print("P-value:", p)

if p < 0.05:
    print("Significant Relationship Found")
else:
    print("No Significant Relationship")

# --------------------------------
# 3. Fare vs Survival
# --------------------------------

survived = df[df["Survived"] == 1]["Fare"]
not_survived = df[df["Survived"] == 0]["Fare"]

t_stat, p = ttest_ind(
    survived,
    not_survived,
    equal_var=False
)

print("\nFare vs Survival")
print("P-value:", p)

if p < 0.05:
    print("Fare has significant effect on survival")
else:
    print("Fare has no significant effect")

print("\nStatistical Analysis Completed!")