import random

import pandas as pd


def get_missing_value(feature, threshold=0.8):
    """
    返回缺失值统计信息

    Args:
        feature: 待处理特征数据框
        threshold: 缺失值阈值

    Returns:
        missing_value：缺失值统计信息
    """
    missing_value = {'drop': [], 'result': {}}
    lens = feature.shape[0]
    for c in feature:
        sm = feature[c].describe()
        missing_rate = round(1 - sm['count'] / lens, 2)
        missing_value['result'][c] = {'missing_rate': missing_rate}
        if missing_rate > threshold:
            missing_value['drop'].append(c)
        print(f"{c} - missing_value: {missing_rate}")
    missing_value['drop_number'] = len(missing_value['drop'])
    return missing_value


def get_unique_value(feature, threshold=0.9):
    """
    返回唯一率统计信息

    Args:
        feature: 待处理特征数据框
        threshold: 唯一率阈值

    Returns:
        unique_value：唯一率统计信息
    """
    unique_value = {'drop': [], 'result': {}}
    category_var = feature.select_dtypes(include=['category']).columns
    for c in category_var:
        sm = feature[c].describe()
        unique_rate = round(sm['unique']/sm['count'], 2)
        unique_value['result'][c] = {"unique_rate": unique_rate}
        if unique_rate > threshold:
            unique_value['drop'].append(c)
        print(f"{c} - unique_value: {unique_value}")
    unique_value['drop_number'] = len(unique_value['drop'])
    return unique_value


def get_freq_value(feature, threshold=0.9):
    """
    返回众数比率统计信息

    Args:
        feature: 待处理特征数据框
        threshold: 众数比率阈值

    Returns:
        freq_value：众数比率统计信息
    """
    freq_value = {'drop': [], 'result': {}}
    category_var = feature.select_dtypes(include=['category']).columns
    for c in category_var:
        sm = feature[c].describe()
        freq_rate = round(sm['freq']/sm['count'], 2)
        freq_value['result'][c] = {"freq_rate": freq_rate}
        if freq_rate > threshold:
            freq_value['drop'].append(c)
        print(f"{c} - freq_value: {freq_value}")
    freq_value['drop_number'] = len(freq_value['drop'])
    return freq_value


def get_variance_value(feature, threshold=0):
    """
    返回方差值统计信息

    Args:
        feature: 待处理特征数据框
        threshold: 方差阈值

    Returns:
        variance_value：方差统计信息
    """
    variance_value = {'drop': [], 'result': {}}
    numeric_var = feature.select_dtypes(include=['float']).columns
    for c in numeric_var:
        sm = feature[c].describe()
        variance = round(sm['std'] ** 2, 2)
        variance_value['result'][c] = {"variance": variance}
        if variance <= threshold or pd.isna(variance):
            variance_value['drop'].append(c)
        print(f"{c} - variance_value: {variance}")
    variance_value['drop_number'] = len(variance_value['drop'])
    return variance_value


def iv_compare(iv_result, var1, var2):
    """
    返回特征var1, var2中IV较大的变量名
    Args:
        iv_result: IV值统计信息
        var1: 变量名1
        var2: 变量名2

    Returns:
        IV较大的变量名
    """
    if var1 in iv_result.keys() and var2 in iv_result.keys():
        var1_iv = iv_result[var1]
        var2_iv = iv_result[var2]
        if var1_iv > var2_iv:
            return var1
        else:
            return var2
    elif var1 not in iv_result.keys() and var2 not in iv_result.keys():
        return random.choice([var1, var2])
    else:
        return None


def get_cor_drop(feature, iv_result, threshold=0.7):
    """
    返回相关性又高，IV值相对也高德变量列表
    Args:
        feature: 待处理数据框
        iv_result: IV值信息
        threshold: 相关系数缺失阈值
    Returns:
        cor_drop: 相关性与IV值相对较高的变量列表
    """
    numeric_var = feature.select_dtypes(include=['float']).columns
    cor = feature[numeric_var].corr().apply(abs)
    cor = cor[(cor > threshold) & (cor != 1)].dropna(how='all', axis=1).dropna(how='all')
    if not cor.empty:
        cor = cor.stack().reset_index()
        cor.columns = ['Var_1', 'Var_2', 'cor']
        cor.drop_duplicates(subset=['cor'], inplace=True)
        cor.sort_values(by='cor', ascending=False, inplace=True)
        cor['smaller_iv'] = cor.apply(lambda x: iv_compare(iv_result, x['Var_1'], x['Var_2']), axis=1)
        cor_drop = list(filter(lambda x: x, set(cor['smaller_iv'].values)))
        return cor_drop
    else:
        return []
