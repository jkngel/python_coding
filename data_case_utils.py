### General import statement
# Data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


### Exploration
data.describe(include='all')
data.corr()[col]
#ex. train.corr()["duration"]


### Plotting
# Histogram
sns.distplot

# Box-plot
sns.boxplot

### Check missing data
data.isnull().sum()
# If many missing data, check where in the dataset they appear: (if appear together, can used dropna())
data[data.isnull()]


### Dataset Processing
# Date
from datetime import datetime
train["year"] = train["start_time"].dt.year
train["month"] = train["start_time"].dt.month
train["day"] = train["start_time"].dt.day
train["weekday"] = train["start_time"].dt.weekday_name
train["hour"] = train["start_time"].dt.hour
train["week_of_year"] = train["start_time"].dt.weekofyear
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

### One-hot encoding
categorical_features = []# ["year","month","weekday","hour"]
X = pd.get_dummies(data.drop(columns=["year","month","weekday","hour"]))
pd.get_dummies(df, prefix=['col1', 'col2'])
    
    
### Sampling
t1 = train.sample(frac=0.001, replace=False)


### Model training and testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_processed, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_true_tmp, y_pred_tmp = y_test, clf.predict(X_test)

### RandomForest feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
# Print the feature ranking
print("Feature ranking:")
feature_columns = features_processed.columns 
for f in range(features_processed.shape[1]):
    print("%d. %s (%f)" % (f + 1, feature_columns[indices[f]], importances[indices[f]]))
# Plot
top_n_features, top_n_feature_strengths = feature_columns[indices], importances[indices]
sns.set_style("dark")
sns.set(rc={'figure.figsize':(10,8)})
plt.barh(top_n_features, top_n_feature_strengths)

