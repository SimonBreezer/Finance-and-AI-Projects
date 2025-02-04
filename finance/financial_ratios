import pandas as pd

# Load dataset
df = pd.read_csv("financial_ratios.csv")

# Calculate financial ratios
df["Gross_Profit_Margin"] = (df["Revenue"] - df["COGS"]) / df["Revenue"]
df["Operating_Profit_Margin"] = df["Operating_Profit"] / df["Revenue"]
df["Net_Profit_Margin"] = df["Net_Profit"] / df["Revenue"]
df["Current_Ratio"] = df["Current_Assets"] / df["Current_Liabilities"]

# Save results to a new CSV
df.to_csv("financial_ratios_calculated.csv", index=False)

# Display results
print(df[["Date", "Gross_Profit_Margin", "Operating_Profit_Margin", "Net_Profit_Margin", "Current_Ratio"]])
