import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from mksc import config
from mksc.utils.reader import read_file


def load_data(mode, read_local):
    """
    加载配置文件指定数据源，返回数据
    Args:
        mode: 数据集读取类别。
            --“train”: 读取带标签的训练数据集
            --“predict": 读取不带标签的预测数据集
        read_local: 是否读取本地文档
    Returns:
        data: 配置文件数据框
    """
    assert mode in ["train", "predict"], "mode only accept [train/predict]"
    assert isinstance(read_local, bool), "read_local is boolean type"
    if mode == 'train':
        sql = config.get('DATABASE', 'TRAIN_SQL')
        filename = config.get('PATH', 'TRAIN_DATASET')
        uri = config.get('DATABASE', 'TRAIN_ENGINE_URL')
    elif mode == "predict":
        sql = config.get('DATABASE', 'PREDICT_SQL')
        filename = config.get('PATH', 'PREDICT_DATASET')
        uri = config.get('DATABASE', 'PREDICT_ENGINE_URL')
    else:
        raise ValueError("Wrong mode type passed, only accepted [train/predict]")

    assert sql or filename, "远程路径与本地路径至少需要提供一个"

    pickle_file = os.path.join(os.getcwd(), "data", f"{mode}.pickle")
    filename = pickle_file if os.path.exists(pickle_file) else filename

    if read_local:
        print(f"Warning: 若存在本地相应文件{mode}.pickle, 将直接读取该文件")
        data = read_file(filename)
    else:
        assert sql and uri, "请完成远程配置"
        engine = create_engine(uri)
        data = pd.read_sql_query(sql, engine)

    # 大小写标准化
    data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

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
    variable_type = pd.read_csv("config/variable_type.csv", encoding='utf_8_sig')
    type_ = 'Type:[identifier/numeric/category/datetime/text/prediction]'
    is_save = 'isSave:[0/1]'

    def get_type_columns(t):
        name = variable_type[(variable_type[type_] == t) & (variable_type[is_save] == 1)]
        return list(name["Variable"])

    y_var = get_type_columns('prediction')[0]
    numeric_var = get_type_columns('numeric')
    category_var = get_type_columns('category')
    datetime_var = get_type_columns('datetime')
    identifier_var = get_type_columns('identifier')
    text_var = get_type_columns('text')
    return numeric_var, category_var, datetime_var, y_var, identifier_var, text_var


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


def replace_default_to_nan(df):
    default_values = pd.read_csv("config/variable_type.csv", dtype={"Default": str}, encoding='utf_8_sig')
    for c in df:
        if default_values[default_values['Variable'] == c].iloc[0, 2] != "prediction":
            default_value = default_values[default_values['Variable'] == c]["Default"].values[0]
            if not pd.isna(default_value):
                if df[c].dtypes == 'object':
                    df[c] = df[c].replace(default_value, np.nan)
                elif df[c].dtypes == 'float':
                    df[c] = df[c].replace(float(default_value), np.nan)
                elif df[c].dtypes == 'int':
                    df[c] = df[c].replace(float(default_value), np.nan)
                    df[c] = df[c].replace(int(default_value), np.nan)
    return df
