import numpy as np

from mksc.feature import binning


def transform(feature, feature_engineering, woe=True):
    """
    特征按照特征工程中保存的结果进行转化
    Args:
        feature: 待转化的特征表
        feature_engineering:特征工程元数据表

    Returns:
        已转化的特征表

    """
    # 缺失值处理
    missing_filling = feature_engineering['missing_filling']
    for c in set(feature.columns) & set(missing_filling['fixed']):
        fill_value = missing_filling['result'][c]['fill_value']
        feature[c].fillna(fill_value, inplace=True)

    # 极端值处理
    abnormal_value = feature_engineering['abnormal_value']
    for c in set(feature.columns) & set(abnormal_value['fixed']):
        max_ = abnormal_value['result'][c]['max']
        min_ = abnormal_value['result'][c]['min']
        feature.loc[:, c] = feature.loc[:, c].apply(lambda x: x if (x < max_) & (x > min_) else np.nan)

    # 标准化处理
    scale_result = feature_engineering['scale_result']['result']
    for c in set(feature.columns) & set(scale_result.keys()):
        mean = scale_result[c]['mean']
        std = scale_result[c]['std']
        feature.loc[:, c] = feature.loc[:, c].apply(lambda x: (x - mean)/std if x else x)

    # 正态化处理
    # standard_lambda = feature_engineering['standard_lambda']
    # if standard_lambda:
    #     for c in set(feature.columns) & set(standard_lambda.keys()):
    #         _lambda = standard_lambda[c]
    #         feature.loc[:, c] = feature.loc[:, c] + 0.5
    #         feature.loc[:, c] = feature.loc[:, c].apply(lambda x: (x**_lambda - 1) / _lambda if _lambda > 0 else log(x))

    # woe转化
    if woe:
        woe_result = feature_engineering['woe_result']
        bin_result = feature_engineering['bin_result']
        feature = binning.woe_transform(feature, woe_result, bin_result)

    return feature
