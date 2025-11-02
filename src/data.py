from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
heart_disease = fetch_ucirepo(id=45)
X_raw = heart_disease.data.features
y_raw = heart_disease.data.targets

# Perform all the cleaning steps
heart_disease_df = pd.concat([X_raw, y_raw], axis=1).dropna()

# Define the final, clean X and y that will be imported
X = heart_disease_df.drop(['num', 'chol'], axis=1)
y_regression = heart_disease_df['chol']         # regression target
y_classification = heart_disease_df['num']      # classification target
print(heart_disease.metadata, X_raw)

# variable information
print(heart_disease.variables, y_raw)
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

df.to_csv("data/cleaned_cleveland.csv", index=False)

print(df.head())
print(df['presence'].value_counts())


