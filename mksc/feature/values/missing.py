import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype


def fix_missing_value(feature, threshold=0.1):
    """
    修正数据框中的数值型变量中的缺失值

    Args:
        feature: 全量数据框
        threshold: 需要补缺失值的列
    """
    missing_filling = {'fixed': [], 'result': {}}
    lens = feature.shape[0]
    for c in feature:
        sm = feature[c].describe()
        missing_rate = round(1 - sm['count'] / lens, 2)
        missing_filling['result'][c] = {'missing_rate': missing_rate}
        if missing_rate < threshold:
            if isinstance(feature[c].dtype, CategoricalDtype):
                fill_value = sm['top']
            elif isinstance(feature[c].dtype, np.dtype):
                fill_value = round(sm['mean'], 4)
            else:
                fill_value = np.nan
            feature[c].fillna(fill_value, inplace=True)
            missing_filling["fixed"].append(c)
            missing_filling['result'][c]['fill_value'] = fill_value

        print(f"{c} - missing_rate: {missing_rate}")
    return feature, missing_filling
