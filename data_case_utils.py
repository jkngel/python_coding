
# Exploration
data.describe()

data.corr()[col]
#ex. train.corr()["duration"]

# One-hot encoding
categorical_features = []# ["year","month","weekday","hour"]

for col in categorical_features:
    tmp_dummies = pd.get_dummies(X_train[col], prefix=col)
    features_processed = pd.concat([features_processed, tmp_dummies], axis=1)
    features_processed = features_processed.drop(col,axis=1)
