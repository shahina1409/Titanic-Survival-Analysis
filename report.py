from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet

# Create PDF
pdf = SimpleDocTemplate("Titanic_Report.pdf")

styles = getSampleStyleSheet()

content = []

# Title
content.append(
    Paragraph(
        "Titanic Survival Analysis Report",
        styles["Title"]
    )
)

content.append(Spacer(1,12))

# Introduction
content.append(
    Paragraph(
        "This project analyzes the Titanic dataset "
        "using Data Science techniques.",
        styles["BodyText"]
    )
)

content.append(Spacer(1,12))

# Dataset Overview
content.append(
    Paragraph(
        "Dataset contains passenger details such as "
        "age, gender, class and fare.",
        styles["BodyText"]
    )
)

content.append(Spacer(1,12))

# Findings
content.append(
    Paragraph(
        "Key Findings:",
        styles["Heading2"]
    )
)

content.append(
    Paragraph(
        "- Female passengers had higher survival rates.",
        styles["BodyText"]
    )
)

content.append(
    Paragraph(
        "- First class passengers survived more frequently.",
        styles["BodyText"]
    )
)

content.append(
    Paragraph(
        "- Higher fare passengers showed better survival chances.",
        styles["BodyText"]
    )
)

content.append(Spacer(1,12))

# Conclusion
content.append(
    Paragraph(
        "Conclusion: Gender, passenger class and fare "
        "were important survival factors.",
        styles["BodyText"]
    )
)

# Build PDF
pdf.build(content)

print("✅ Titanic_Report.pdf Created Successfully!")