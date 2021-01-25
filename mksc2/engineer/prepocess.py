import os
import pandas as pd
import numpy as np
from mksc2.utils.reader import read
from mksc2.utils.reader import config


def load_data(mode):
    """
    加载配置文件指定数据源，返回数据
    Args:
        mode: 数据集读取类别。
            --“train”: 读取带标签的训练数据集
            --“predict": 读取不带标签的预测数据集
    Returns:
        data: 配置文件数据框
    """
    assert mode in ["train", "predict"], "mode only accept [train/predict]"
    if mode == 'train':
        sql = config.get('DATABASE', 'TRAIN_SQL')
        filename = config.get('PATH', 'TRAIN_DATASET')
        engine = config.get('DATABASE', 'TRAIN_ENGINE_URL')
    elif mode == "predict":
        sql = config.get('DATABASE', 'PREDICT_SQL')
        filename = config.get('PATH', 'PREDICT_DATASET')
        engine = config.get('DATABASE', 'PREDICT_ENGINE_URL')
    else:
        raise ValueError("Wrong mode type passed, only accepted [train/predict]")

    assert sql or filename, "远程路径与本地路径至少需要提供一个"

    if os.path.exists(os.path.join(os.getcwd(), "data", f"{mode}.pickle")):
        print(f"Warning: 存在本地相应文件{mode}.pickle, 将直接读取该文件")
        filename = os.path.join(os.getcwd(), "data", f"{mode}.pickle")

    try:
        data = read(filename)
    except:
        data = read(sql, engine)

    # 大小写标准化
    for c in data.select_dtypes(include=['object']).columns:
        data[c] = data[c].str.lower()

    # 空值标准化
    data.replace("", np.nan, inplace=True)
    data.replace("null", np.nan, inplace=True)
    data.replace("none", np.nan, inplace=True)
    data.replace("na", np.nan, inplace=True)
    return data


def get_variable_type():
    """
    根据指定变量类型的配置表，返回各类型的变量列表

    Returns:
        numeric: 数值型变量列表
        category: 类别型变量列表
        datetime: 日期型变量列表
        label_name: 标签列
        id: 唯一标识列
    """
    variable_type = pd.read_csv("config/variable_type.csv", encoding='gbk')
    label_var = variable_type[variable_type.iloc[:, 2] == 'label']
    numeric_var = variable_type[(variable_type.iloc[:, 2] == 'numeric') & (variable_type.iloc[:, 1] == 1)]
    category_var = variable_type[(variable_type.iloc[:, 2] == 'category') & (variable_type.iloc[:, 1] == 1)]
    datetime_var = variable_type[(variable_type.iloc[:, 2] == 'datetime') & (variable_type.iloc[:, 1] == 1)]
    label_name = label_var['Variable'].tolist()[0]
    numeric = list(numeric_var['Variable'])
    category = list(category_var['Variable'])
    datetime = list(datetime_var['Variable'])
    return numeric, category, datetime, label_name

def variable_classify(feature):
    """
    对数据框feature的变量进行分类，返回各类别的变量列表

    Args:
        feature: 待分类的数据框

    Returns:
        numeric_var: 数值型变量列表
        category_var: 类别性变量列表
        datetime_var: 日期型变量列表
    """
    numeric_var = feature.select_dtypes(exclude=['object', 'datetime']).columns
    category_var = feature.select_dtypes('object').columns
    datetime_var = feature.select_dtypes('datetime').columns
    return numeric_var, category_var, datetime_var