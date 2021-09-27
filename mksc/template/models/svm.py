import pickle
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC

from mksc.step.feature import processed_feature
from mksc.utils.tools import plot_ks

# 常数参数
metric = "recall"
cv = 10
model_name = 'SVM'
model_path = f"result/model/{model_name}_{str(datetime.now().strftime('%Y%m%d%H%M%S'))}.pickle"

# 加载数据
feature, y = processed_feature(mode="train", do_transform=True, woe=True)
feature = pd.concat([feature, pd.get_dummies(feature[feature.select_dtypes('category').columns])], axis=1)
feature.drop(feature.select_dtypes('category').columns, axis=1, inplace=True)

# 重采样
# feature, y = SMOTE().fit_resample(feature, y)
x_train, x_test, y_train, y_test = train_test_split(feature, y, test_size=0.2, random_state=0)

# 基础模型
model = SVC(probability=True, C=0.6, gamma=0.9)
param = {
        'kernel': ['linear', 'rbf'],
        'C': list(i/10 for i in range(1, 11)),
        'gamma': list(i/10 for i in range(1, 11))
        }

# 随机搜索
model = RandomizedSearchCV(model, param_distributions=param, cv=cv, return_train_score=True, scoring=metric)
model.fit(x_train, y_train)
model = model.best_estimator_
plot_ks(model, x_train, y_train)

# 泛化模型评估
proba_train = model.predict_proba(x_train)
proba_test = model.predict_proba(x_test)
for threshold in list(i/100 for i in range(1, 100))[::-1]:
    print(threshold)
    predict_train = [0 if i[1] < threshold else 1 for i in proba_train]
    predict_test = [0 if i[1] < threshold else 1 for i in proba_test]

    acu_train = accuracy_score(y_train, predict_train)
    acu_test = accuracy_score(y_test, predict_test)

    sen_train = recall_score(y_train, predict_train, pos_label=1)
    sen_test = recall_score(y_test, predict_test, pos_label=1)

    spe_train = recall_score(y_train, predict_train, pos_label=0)
    spe_test = recall_score(y_test, predict_test, pos_label=0)

    print(f'模型准确率：训练 {acu_train * 100:.2f}%	测试 {acu_test * 100:.2f}%     {acu_train - acu_test:.2f}')
    print(f'正例覆盖率：训练 {sen_train * 100:.2f}%	测试 {sen_test * 100:.2f}%     {sen_train - sen_test:.2f}')
    print(f'负例覆盖率：训练 {spe_train * 100:.2f}%	测试 {spe_test * 100:.2f}%     {spe_train - spe_test:.2f}')
    print(f'f1：训练 {2/((1/sen_train) + (1/spe_train)) * 100:.2f}%	测试 {2/((1/sen_test) + (1/spe_test)) * 100:.2f}%')

threshold = "最优阈值"

# 结果保存
result = model, threshold
with open(model_path, 'wb') as f:
    f.write(pickle.dumps(result))
