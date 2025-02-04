import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("financial_data.csv")

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Plot data
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Revenue'], marker='o', label='Revenue', linestyle='-')
plt.plot(df['Date'], df['Expenses'], marker='s', label='Expenses', linestyle='--')
plt.plot(df['Date'], df['Profit'], marker='^', label='Profit', linestyle=':')

# Labels and title
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.title('Financial Dashboard')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
