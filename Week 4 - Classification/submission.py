import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Question 1
train_df_titanic = pd.read_csv('./train_preprocessed.csv')
print(train_df_titanic[['Sex', 'Survived']]. \
      groupby(['Sex']). \
      mean().sort_values(by='Survived', ascending=False))

# Glass data set preprocessing
train_df = pd.read_csv('./glass.csv')
train_df_features = train_df.drop('Type', axis=1)


for column_name in train_df_features.columns:
    print("Correlation for: ", column_name)
    print(train_df[['Type', column_name]]. \
          groupby(['Type']). \
          mean().sort_values(by=column_name, ascending=False))

cols_to_drop = []
for column_name in train_df_features.columns:
    print("Correlation for: ", column_name)
    print(train_df[['Type', column_name]]. \
          groupby(['Type']). \
          mean().sort_values(by=column_name, ascending=False))

print("Silicon should be dropped...")

# Question 2
train_df = pd.read_csv('./glass.csv')
train_df = train_df.drop("Si", axis=1)
# train_df = train_df.drop("Ca", axis=1)
train_df = train_df.drop("RI", axis=1)
X = train_df.drop("Type", axis=1)
y = train_df["Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
acc_gnb = round(gnb.score(X_test, y_test) * 100, 2)
print("nb accuracy is:", acc_gnb)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

print(classification_report(y_test, y_pred))

# Question 3
svc = SVC()
svc.fit(X_train, y_train)
y_pred_2 = svc.predict(X_test)
acc_svc = round(svc.score(X_test, y_test) * 100, 2)
print("svm accuracy is:", acc_svc)

print(classification_report(y_test, y_pred_2))
