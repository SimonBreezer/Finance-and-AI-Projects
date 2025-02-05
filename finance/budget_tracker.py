import pandas as pd

# Load expenses dataset
df = pd.read_csv("finance/expenses.csv")

# Define monthly budget for each category
budget = {
    "Housing": 1500,
    "Food": 500,
    "Utilities": 150,
    "Transportation": 100,
    "Entertainment": 100
}

# Group by category and calculate total expenses
category_totals = df.groupby("Category")["Amount"].sum()

# Compare actual spending vs budget
budget_status = {category: category_totals.get(category, 0) - budget.get(category, 0) for category in budget}

# Print results
print("Budget Status Report:")
for category, status in budget_status.items():
    if status > 0:
        print(f"⚠️ Over budget in {category} by ${status:.2f}")
    else:
        print(f"✅ Under budget in {category} by ${-status:.2f}")

# Save results to a CSV
budget_report = pd.DataFrame.from_dict(budget_status, orient="index", columns=["Budget Status"])
budget_report.to_csv("finance/budget_report.csv")
