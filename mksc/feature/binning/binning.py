from math import log

import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def tree_binning(y, feature, positive_tag=1, max_bin_num=6):
    """
    二分类
    通过决策树执行模型分箱函数
     y_i: 第i组中1的个数
     y_T: 所有1的个数
     n_i: 第i组中0的个数
     n_T: 所有0的个数
     WOE_i = ln(P_yi/P_ni) = ln(y_i*n_T/y_T/n_i)
     IV = sum(IV_i) = sum((P_yi-P_ni)*WOE_i) = sum((y_i/y_T-n_i/n_T)*WOE_i)

    Args:
        y: 待分箱特征的标签
        feature: 待分箱特征的数据框
        positive_tag: pass
        max_bin_num: pass
    Returns:
        bin_result: 分箱结果数据
        iv_result: IV值统计结果
        woe_result: WOE值统计结果
        woe_adjust_result: 需要调整分箱的结果
    """
    assert y.nunique() == 2
    negative_tag = list(filter(lambda x: x != positive_tag, y.unique()))[0]
    bin_result = {'result': {}, 'na_error': [], 'woe_adjust_result': []}
    iv_result = {}
    woe_result = {}

    y_name = y.name
    y_t = y[y.values == positive_tag].shape[0]
    n_t = y[y.values == negative_tag].shape[0]
    numeric_var = feature.select_dtypes(include=['float']).columns
    for c in numeric_var:
        temp = pd.concat([feature[c], y], axis=1)
        # 统计空值分布
        binning_data_na = temp[temp[c].isna()]
        y_na = binning_data_na[y_name][binning_data_na[y_name].values == positive_tag].shape[0]
        n_na = binning_data_na[y_name][binning_data_na[y_name].values == negative_tag].shape[0]
        df_na = pd.DataFrame([['nan', y_na, n_na]], columns=[c, 'y_i', 'n_i'])

        # cart树确定分箱边界
        binning_data = temp.dropna(subset=[c])
        if binning_data.empty:
            boundary = []
        else:
            model = DecisionTreeClassifier(min_samples_leaf=30, max_depth=5, max_leaf_nodes=max_bin_num)
            model.fit(binning_data[c].values.reshape(-1, 1), binning_data[y_name].values)
            boundary = model.tree_.threshold
            boundary = sorted(list(set([round(i, 4) for i in boundary[boundary != -2]])))
            boundary = [float('-inf')] + boundary + [float('inf')]

        bin_result['result'][c] = boundary
        # 统计分箱结果
        df = pd.merge(pd.cut(temp[c], boundary), temp[y_name], right_index=True, left_index=True)
        df['count'] = 1
        df = df.groupby([c, y_name]).count().unstack(level=1)
        df.fillna(0, inplace=True)
        df.columns = [x[0] + '_' + str(x[1]) for x in df.columns.values]
        df.rename(columns={f'count_{negative_tag}': 'n_i', f'count_{positive_tag}': 'y_i'}, inplace=True)
        df.reset_index(inplace=True)
        df = df.append(df_na, ignore_index=True)
        df['woe_i'] = df.apply(lambda x: log(x['y_i'] + 0.5) + log(n_t) - log(y_t) - log(x['n_i'] + 0.5), axis=1)
        df['iv_i'] = df.apply(lambda x: (x['y_i'] / y_t - x['n_i'] / n_t) * x['woe_i'], axis=1)

        woe_result[c] = df[[c, 'woe_i']]
        iv_result[c] = df['iv_i'].sum()
        print(f"{c} - tree_binning: {boundary}")

    return bin_result, iv_result, woe_result


def uniform_binning(y, feature, positive_tag=1, max_bin_num=6):
    assert y.nunique() == 2
    negative_tag = y.unique().remove_categories(positive_tag)[0]
    bin_result = {'result': {}, 'na_error': [], 'woe_adjust_result': []}
    iv_result = {}
    woe_result = {}

    y_name = y.name
    y_t = y[y.values == positive_tag].shape[0]
    n_t = y[y.values == negative_tag].shape[0]
    numeric_var = feature.select_dtypes(include=['float']).columns
    for c in numeric_var:
        temp = pd.concat([feature[c], y], axis=1)
        # 统计空值分布
        binning_data_na = temp[temp[c].isna()]
        y_na = binning_data_na[y_name][binning_data_na[y_name].values == positive_tag].shape[0]
        n_na = binning_data_na[y_name][binning_data_na[y_name].values == negative_tag].shape[0]
        if y_na and n_na:
            df_na = pd.DataFrame([['nan', y_na, n_na]], columns=[c, 'y_i', 'n_i'])
        elif (y_na and not n_na) or (not y_na and n_na):
            bin_result['na_error'].append({c: positive_tag if y_na else negative_tag})
            continue
        else:
            df_na = pd.DataFrame(columns=[c, 'y_i', 'n_i'])

        #
        binning_data = temp.dropna(subset=[c])
        max_ = binning_data[c].max()
        min_ = binning_data[c].min()
        width = (max_ - min_) / (max_bin_num - 2)
        boundary = [i * width for i in range(max_bin_num - 1)]
        boundary = [float('-inf')] + boundary + [float('inf')]
        bin_result['result'][c] = boundary

        # 统计分箱结果
        df = pd.merge(pd.cut(temp[c], boundary), temp[y_name], right_index=True, left_index=True)
        df['count'] = 1
        df = df.groupby([c, y_name]).count().unstack(level=1)
        df.columns = [x[0] + '_' + str(x[1]) for x in df.columns.values]
        df.rename(columns={f'count_{negative_tag}': 'n_i', f'count_{positive_tag}': 'y_i'}, inplace=True)
        df.reset_index(inplace=True)

        temp2 = df[df.isnull().T.any()]
        if not temp2.empty:
            y_i = temp2['y_i'].values[0]
            bin_result['woe_adjust_result'].append({c: {"bins": temp2[c].values[0], "category": negative_tag if pd.isna(y_i) else positive_tag}})

        df = df.append(df_na, ignore_index=True)
        df['woe_i'] = df.apply(lambda x: log(x['y_i']) + log(n_t) - log(y_t) - log(x['n_i']), axis=1)
        df['iv_i'] = df.apply(lambda x: (x['y_i'] / y_t - x['n_i'] / n_t) * x['woe_i'], axis=1)
        woe_result[c] = df[[c, 'woe_i']]
        iv_result[c] = df['iv_i'].sum()

    return bin_result, iv_result, woe_result
