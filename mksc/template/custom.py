import featuretools as ft
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from mksc.core.prepocess import load_data, get_variable_type, replace_default_to_nan


class Custom(object):
    """
    自定义预处理类函数封装
    此处可以记录模型评估与调整过程
    """

    target = "classify"
    cross = False
    adjust_bins = [float('-inf'), 100, 200, 300, float('inf')]

    def clean_data(self, feature, y):
        """
        基于单元格与行操作的数据清洗

        Args:
            feature: 待清洗特征数据框
            label: 待清洗标签序列

        Returns:
            Feature: 已清洗特征数据框
            Label: 已清洗标签序列
        """
        feature = replace_default_to_nan(feature)
        # 分类任务
        y = y.astype("category") if self.target == "classify" else y.astype("float")
        y = y.map(lambda x: int(x)) if self.target == "classify" else y
        # ------------------------------------------
        # 自定义值清洗, 不改变列
        # ------------------------------------------
        return feature, y

    def feature_combination(self, feature, y):
        """
        基于列操作的数据清洗与特征构造

        Args:
            feature: 待清洗特征数据框

        Returns:
            feature_tmp: 已清洗特征数据框
        """
        category_var = feature.select_dtypes(include='category').columns
        # ------------------------------------------
        # 构造衍生变量, 建议最后保存清洗列至类变量used_var
        # eg: feature_tmp['new_variable'] = feature_tmp['old_variable']
        #     feature_tmp.drop('old_variable', axis=1, inplace=True)
        #     self.used_var = ['old_variable']
        # ------------------------------------------
        # 2 阶交叉
        if self.cross:
            tag = None
            while tag != "success":
                try:
                    es = ft.EntitySet(id='single_dataframe')
                    es.entity_from_dataframe(entity_id='custom_feature', dataframe=feature, index="index")
                    feature, feature_names = ft.dfs(entityset=es, target_entity='custom_feature', max_depth=1, trans_primitives=['multiply_numeric', 'divide_numeric'])
                    tag = "success"
                except MemoryError:
                    print("MemoryError samples squeeze 10%")
                    feature.drop("index", axis=1, inplace=True)
                    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
                    for new, drop in split.split(feature, y):
                        feature = feature.loc[new]
                        y = y.loc[new]
                    feature.reset_index(inplace=True, drop=True)
                    y.reset_index(inplace=True, drop=True)

        # 日期变量处理, 提取完后需要丢弃变量
        datetime_var = feature.select_dtypes(include='datetime64[ns]').columns
        text_var = feature.select_dtypes(include='object').columns
        num_var = feature.select_dtypes(include='float').columns
        # date_var = ['day', 'dayofweek', 'dayofyear', 'days_in_month', 'is_leap_year', 'is_month_end',
        #             'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end',
        #             'is_year_start', 'month', 'quarter', 'week', 'weekday', 'weekofyear', 'year']
        # for dv in datetime_var:
        #     for v in date_var:
        #         feature[f"{dv}__{v}"] = eval(f"feature['{dv}'].dt.{v}")
        #         feature[f"{dv}__{v}"] = feature[f"{dv}__{v}"].astype("category")
        feature.drop(list(datetime_var) + list(text_var), axis=1, inplace=True)
        feature[num_var] = feature[num_var].applymap(lambda x: round(x, 4))
        feature[num_var] = feature[num_var].applymap(lambda x: np.nan if x == float("inf") or x == float("-inf") else x)
        feature[category_var] = feature[category_var].astype('category')
        return feature


if __name__ == "__main__":
    # 自定义调试用代码
    # 加载数据、变量类型划分、特征集与标签列划分
    data = load_data(mode="train", read_local=True)
    data = replace_default_to_nan(data)
    numeric_var, category_var, datetime_var, y_var, identifier_var, text_var = get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    y = data[y_var]
    try:
        feature[numeric_var] = feature[numeric_var].astype('float')
        feature[category_var] = feature[category_var].astype('category')
        feature[datetime_var] = feature[datetime_var].astype('datetime64[ns]')
    except Exception as e:
        print(e)
