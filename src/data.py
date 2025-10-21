import pandas as pd

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "presence"
]

df = pd.read_csv("Data/processed.cleveland.data", names=columns, na_values='?')

#dropped all the rows with missing vales
df = df.dropna()

#ensures all data is numeric
df = df.apply(pd.to_numeric)

#converted to binary classification
df['presence'] = df['presence'].apply(lambda x: 1 if x > 0 else 0)

print(df.head())
print(df['presence'].value_counts())


