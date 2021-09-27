import numpy as np


def fix_abnormal_value(feature, method='boundary'):
    """
    修正数据框中的数值型变量中的异常值

    Args:
        feature: 待修正的数据框
        method: 补值方法
            -- boundary：缩放到边界
            -- drop: 丢弃
    Returns:
        feature: 已处理数据框
        abnormal_value: 异常值统计结果
    """
    abnormal_value = {"fixed": [], 'result': {}, 'method': method}
    c_var = feature.select_dtypes("float").columns
    for c in c_var:
        if feature[c].count():
            sm = feature[c].describe()
            iqr = round(sm['75%'] - sm['25%'], 4)
            min_ = round(sm['25%'] - 1.5*iqr, 4)
            max_ = round(sm['75%'] + 1.5*iqr, 4)
            abnormal_value_indexes = list(feature.loc[(feature[c] < min_) | (feature[c] > max_)].index)
            abnormal_value_length = len(abnormal_value_indexes)
            abnormal_value_rate = round(abnormal_value_length/feature[c].count(), 3)
            abnormal_value['result'][c] = {'abnormal_value_length': abnormal_value_length,
                                           'abnormal_value_rate': abnormal_value_rate,
                                           'max': max_,
                                           'min': min_}
            abnormal_value['fixed'].append(c)
            if method == "drop":
                feature.loc[:, c] = feature.loc[:, c].apply(lambda x: x if (x < max_) & (x > min_) else np.nan)
            else:
                feature.loc[:, c] = feature.loc[:, c].apply(lambda x: max_ if x > max_ else(min_ if x < min_ else x))
        else:
            abnormal_value_rate = 0
            abnormal_value['result'][c] = {'abnormal_value_length': 0,
                                           'abnormal_value_rate': 0,
                                           'max': np.nan,
                                           'min': np.nan}
        print(f"{c} - abnormal_rate: {abnormal_value_rate}")
    return feature, abnormal_value
