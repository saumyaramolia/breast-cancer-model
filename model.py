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
df.isna().sum()
df = df.dropna(axis=1)
df.head(10)
lb = LabelEncoder()
df.iloc[:, 1] = lb.fit_transform(df.iloc[:, 1].values)
df['diagnosis'].value_counts()
plt.figure(figsize=(25, 25))
sns.heatmap(df.iloc[:, 1:10].corr(), annot=True)
sns.pairplot(df.iloc[:, 1:5], hue="diagnosis")
X = df.iloc[:, 2:32].values
y = df.iloc[:, 1].values.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
st = StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.fit_transform(X_test)
print(X_train.shape)
log = LogisticRegression()
log.fit(X_train, y_train)
print(log.score(X_train, y_train) * 100)
accuracy_score(y_test, log.predict(X_test))
print(classification_report(y_test, log.predict(X_test)))
pickle.dump(log, open("./model/model.pkl", "wb"))
