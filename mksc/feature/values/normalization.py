def normalization(feature):
    """
    对数据框中的数据进行最大最小值标准化

    Args:
        feature: 待处理的数据框
    Returns:
        feature: 已处理数据框
        normalization_result: 归一化结果集
    """
    numeric_var = feature.select_dtypes(exclude=['object', 'datetime']).columns
    normalization_result = {"result": {}}
    for c in numeric_var:
        sm = feature[c].describe()
        normalization_result[c] = {'max': sm['max'], 'min': sm['min']}
        feature[c] = feature[c].apply(lambda x: (x - sm['min'])/(sm['max']-sm['min']) if x else x)
    return feature, normalization_result
