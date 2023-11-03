import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv('data.csv')
df.head(10)

# Drop columns that are not needed
df = df[['area_mean','area_se', 'concavity_mean', 'concavity_se', 'concavity_worst', 'fractal_dimension_se',
         'fractal_dimension_worst', 'smoothness_worst', 'symmetry_worst', 'texture_mean', 'diagnosis']]

lb = LabelEncoder()
df['diagnosis'] = lb.fit_transform(df['diagnosis'].values)
df['diagnosis'].value_counts()

plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)
sns.pairplot(df, hue="diagnosis")

X = df.iloc[:, 0:10].values
y = df['diagnosis'].values.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

st = StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

print(X_train.shape)

log = LogisticRegression()
log.fit(X_train, y_train)

print(log.score(X_train, y_train) * 100)
accuracy_score(y_test, log.predict(X_test))

print(classification_report(y_test, log.predict(X_test)))

# Save the model
pickle.dump(log, open("./model/model.pkl", "wb"))
