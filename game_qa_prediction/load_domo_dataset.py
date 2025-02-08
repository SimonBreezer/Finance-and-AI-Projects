import domojupyter as domo

# Load the dataset
df = domo.read_dataframe('QA ML Prediction Data', query='SELECT * FROM table')

# Set the first row as column headers
df.columns = df.iloc[0]  # Assign first row as column names
df = df[1:]  # Remove the first row (now redundant)

# Reset the index
df.reset_index(drop=True, inplace=True)

# Display first few rows
df.head()
