import pandas as pd


def woe_transform(feature, woe_result, bin_result):
    """
    将特征数据框转换为WOE分箱数据框

    Args:
        feature: 带转换数据框
        woe_result: 分组WOE值结果
        bin_result: 分组边界结果

    Returns:
        feature：已转换数据框
    """
    numeric_var = feature.select_dtypes(exclude=['category']).columns
    bin_var = list(set(numeric_var) & set(woe_result.keys()))
    for c in bin_var:
        woe = woe_result[c]
        feature[c] = pd.cut(feature[c], bin_result['result'][c])
        feature[c] = feature[c].apply(lambda x: x if pd.isna(x) else woe.loc[woe[c] == x, 'woe_i'].values[0])
        feature[c] = feature[c].astype('float')
        feature[c] = feature[c].apply(lambda x: woe.loc[0, 'woe_i'] if pd.isna(x) else x)
    return feature
