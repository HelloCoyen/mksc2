import numpy as np

def fix_scaling(feature):
    """
    对数据框中的数据进行标准差标准化处理

    Args:
        feature: 待处理的数据框
    Returns:
        feature: 已处理数据框
    """
    scale_result = {"fixed": [], "result": {}}
    numeric_var = feature.select_dtypes("float").columns
    for c in numeric_var:
        sm = feature[c].describe()
        mean = round(sm["mean"], 4)
        std = round(sm["std"], 4)
        scale_result['result'][c] = {'mean': mean, 'std': std}
        feature[c] = feature[c].apply(lambda x: round((x - mean)/std, 4) if x else x)
        feature[c] = feature[c].apply(lambda x: np.nan if x == float("inf") or x == float("-inf") else x)
        scale_result["fixed"].append(c)

        print(f"{c} - fix_scaling: mean：{mean} std：{std}")
    return feature, scale_result
