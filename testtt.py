import pandas as pd

df = pd.read_csv('Data/AllMonths/data_cleaned_interface.csv')
print(df.columns)
print(df["site_business_hours"].value_counts())

