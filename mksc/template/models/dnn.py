import pickle
from datetime import datetime

import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers

from mksc.step.feature import processed_feature
from mksc.utils.tools import plot_ks


def dataframe_to_dataset(feature, y, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(feature), pd.get_dummies(y).values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(feature))
    ds = ds.batch(batch_size)
    return ds


# 常数参数
metric = "recall"
cv = 10
model_name = 'DNN'
model_path = f"result/model/{model_name}_{str(datetime.now().strftime('%Y%m%d%H%M%S'))}.pickle"

# 加载数据
feature, y = processed_feature(mode="train", do_transform=True, woe=True)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(feature, y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
print(len(x_train), 'train examples')
print(len(x_val), 'validation examples')
print(len(x_test), 'test examples')

batch_size = 5
train_ds = dataframe_to_dataset(x_train, y_train, batch_size=batch_size)
val_ds = dataframe_to_dataset(x_val, y_val,  shuffle=False, batch_size=batch_size)
test_ds = dataframe_to_dataset(x_test, y_test, shuffle=False, batch_size=batch_size)

feature_columns = []
for c in feature.select_dtypes("float").columns:
    feature_columns.append(feature_column.numeric_column(c))

for c in feature.select_dtypes("category").columns:
    c_var = feature_column.categorical_column_with_vocabulary_list(c, feature[c].unique().to_list())
    c_var_one_hot = feature_column.indicator_column(c_var)
    feature_columns.append(c_var_one_hot)

# 基础模型
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=1)

loss, Recall = model.evaluate(test_ds)
print("Accuracy", Recall)
plot_ks(model, list(x_train.T.to_numpy()), y_train)

# 泛化模型评估
proba_train = model.predict_proba(list(x_train.T.to_numpy()))
proba_test = model.predict_proba(list(x_test.T.to_numpy()))
for threshold in list(i/100 for i in range(1, 100))[::-1]:
    print(threshold)
    predict_train = [0 if i[0] < threshold else 1 for i in proba_train]
    predict_test = [0 if i[0] < threshold else 1 for i in proba_test]

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
