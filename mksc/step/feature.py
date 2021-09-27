import os

import numpy as np
import pandas as pd
from custom import Custom
from statsmodels.iolib.smpickle import load_pickle

from mksc.core.prepocess import load_data, get_variable_type
from mksc.feature import FeatureEngineering
from mksc.feature import transform


def feature_engineering(read_local):
    """
    项目特征工程程序入口
    """
    # 加载数据、变量类型划分、特征集与标签列划分
    print(" >>> 数据加载")
    feature, y = processed_feature(mode="train", do_transform=False, read_local=read_local)
    print(f" >>> 当前数据规模: {feature.shape}\n"
          f"     特征工程预处理数：{feature.shape[1]}\n"
          )

    cs = Custom()

    # 标准化特征工程
    print(">>> 特征工程标准进程")
    fe = FeatureEngineering(feature, y, cs.target)
    fe.run()


def processed_feature(mode, read_local=True, do_transform=True, woe=True):
    """
    模型训练主程序入口
    """
    assert os.path.exists('result/feature_engineering.pickle') if do_transform else True

    data = load_data(mode=mode, read_local=read_local)
    numeric_var, category_var, datetime_var, y_var, identifier_var, text_var = get_variable_type()
    feature = data[numeric_var + category_var + datetime_var + text_var]
    y = data[y_var] if mode == "train" else pd.Series([np.nan]*len(feature), name=y_var)

    # 自定义数据清洗
    cs = Custom()
    feature, y = cs.clean_data(feature, y)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('category')
    feature[datetime_var] = feature[datetime_var].astype('datetime64[ns]')

    # 自定义特征组合，全部为数值变量
    feature = cs.feature_combination(feature, y)

    # 数据处理
    if do_transform:
        feature_engineering = load_pickle('result/feature_engineering.pickle')
        feature = feature[feature_engineering['feature_selected']]
        feature = transform(feature, feature_engineering, woe=woe)
    return feature, y
