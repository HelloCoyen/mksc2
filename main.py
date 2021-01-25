from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier as Model
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
import pickle


data = load_iris()['data']
target = load_iris()['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2021)

model = Model().fit(X_train, y_train)

# train_predict = model.predict(X_train)
# accuracy_score(y_train, train_predict)

# test_predict = model.predict(X_test)
# accuracy_score(y_test, test_predict)

# scores = cross_val_score(model, data, target, cv=5)
# scores.mean()

param = {
    # 'min_split_gain': list(np.linspace(0, 1, num=20)),
    # 'min_child_weight ': list(np.logspace(np.log(0.001), np.log(0.2), base=np.exp(1), num=20)),
    # 'min_child_samples': list(range(10, 500, 30)),
    # 'subsample_for_bin': list(range(20000, 300000, 20000)),
    # 'objective': ['binary', 'multiclass'],
    # 'max_depth': list(range(-1, 30, 5)),
    'boosting_type': ['gbdt', 'goss', 'dart', 'rf'],
    'num_leaves': list(range(10, 200, 10)),
    'learning_rate': list(np.logspace(np.log(0.001), np.log(0.5), base=np.exp(1), num=50)),
    'n_estimators': list(range(10, 500, 50)),
    'class_weight': [None, 'balanced'],
    'subsample': list(i/10 for i in range(1, 11)),
    'colsample_bytree': list(i/10 for i in range(1, 11)),
    'reg_alpha': list(i/10 for i in range(1, 11)),
    'reg_lambda': list(i/10 for i in range(1, 11))
}

# params = {key: random.sample(value, 1)[0] for key, value in param.items()}
# print(params)

# 用RandomSearch+CV选取超参数
random_search = RandomizedSearchCV(model, param_distributions=param, cv=5, return_train_score=True)
random_search.fit(X_train, y_train)
print(random_search.best_params_)
print(random_search.best_score_)

model = random_search.best_estimator_
train_predict = model.predict(X_train)
print(accuracy_score(y_train, train_predict))

test_predict = model.predict(X_test)
print(accuracy_score(y_test, test_predict))

with open("model.pkl", "wb") as f:
    f.write(pickle.dumps(model))