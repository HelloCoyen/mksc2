import math

from scipy.stats import boxcox


def fix_standard(feature):
    """
    对数据框中的数据进行正态化处理

    Args:
        feature: 待处理的数据框

    Returns:
        feature: 已处理数据框
        standard_lambda: 对应特征的lambda值
    """
    numeric_var = feature.select_dtypes(exclude=['category']).columns
    standard_lambda = {}
    for c in numeric_var:
        feature[c], lambda_ = boxcox(feature[c]+0.5)
        standard_lambda[c] = lambda_
    return feature, standard_lambda


def logarithmetics(feature):
    """
    对数变化操作
    Args:
        feature: 待处理的数据框

    Returns:
        feature: 已处理数据框
    """
    numeric_var = feature.select_dtypes(exclude=['object', 'datetime']).columns
    for c in numeric_var:
        feature[c] = feature[c].apply(lambda x: math.log(x) if x else x)
    return feature
