import domojupyter as domo

# Load the dataset
df = domo.read_dataframe('QA ML Prediction Data', query='SELECT * FROM table')

# Display first few rows
df.head()
