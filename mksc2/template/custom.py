import pandas as pd
import numpy as np

class Custom(object):
    """
    自定义预处理类函数封装
    此处可以记录模型评估与调整过程
    """
    def __init__(self):
        self.cleaned_var = []
        self.used_var = []
        self.adjust_var = []
        self.adjust_bins = []

    def clean_data(self, feature, label):
        """
        基于单元格与行操作的数据清洗

        Args:
            feature: 待清洗特征数据框
            label: 待清洗标签序列

        Returns:
            feature_tmp: 已清洗特征数据框
            label_tmp: 已清洗标签序列
        """
        feature_tmp = feature.copy()
        label_tmp = label.copy()
        # ------------------------------------------
        # 自定义值清洗, 建议最后保存清洗列至类变量cleaned_var
        # eg.
        #     feature_tmp['variable'] = feature_tmp['variable']
        #     self.cleaned_var = ['old_variable']
        # ------------------------------------------
        return feature_tmp, label_tmp

    def feature_combination(self, feature):
        """
        基于列操作的数据清洗与特征构造

        Args:
            feature: 待清洗特征数据框

        Returns:
            feature_tmp: 已清洗特征数据框
        """
        feature_tmp = feature.copy()
        # ------------------------------------------
        # 构造衍生变量, 建议最后保存清洗列至类变量used_var
        # eg: feature_tmp['new_variable'] = feature_tmp['old_variable']
        #     feature_tmp.drop('old_variable', axis=1, inplace=True)
        #     self.used_var = ['old_variable']
        # ------------------------------------------

        # 日期变量处理, 提取完后需要丢弃变量
        datetime_var = feature_tmp.select_dtypes(include='datetime64').columns
        date_var = ['day', 'dayofweek', 'dayofyear', 'days_in_month', 'is_leap_year', 'is_month_end',
                    'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end',
                    'is_year_start', 'month', 'quarter', 'week', 'weekday', 'weekofyear', 'year']
        for dv in datetime_var:
            for v in date_var:
                feature_tmp[f"{dv}__{v}"] = eval(f"feature_tmp['{dv}'].dt.{v}")
                feature_tmp[f"{dv}__{v}"] = feature_tmp[f"{dv}__{v}"].astype("object")
        feature_tmp.drop(datetime_var, axis=1, inplace=True)

        # 分类变量one-hot处理，处理完成后需要丢弃变量
        category_var = feature_tmp.select_dtypes(include=['object']).columns
        if not category_var.empty:
            feature_tmp[category_var].fillna("NA", inplace=True)
            one_hot = pd.get_dummies(feature_tmp[category_var], prefix_sep="__")
            one_hot.columns = list(map(lambda x: x.split(".")[0], one_hot.columns))
            feature_tmp = pd.concat([feature_tmp, one_hot], axis=1)
        feature_tmp.drop(category_var, axis=1, inplace=True)
        return feature_tmp

    def feature_adjust(self, feature):
        """
        调整特征表
        Args:
            feature: 原始特征表

        Returns:
            feature_tmp: 调整后的特征数据框
        """
        # ------------------------------------------
        # 后期调整的时候控制某些特征不进入训练，只能减少特征
        # 将人为选择的特征放入类变量adjust_var
        # ------------------------------------------
        feature_tmp = feature[self.adjust_var]
        return feature_tmp

    def show(self):
        """打印全部属性"""
        print(f"本次自定义过程清洗的特征：{self.cleaned_var}")
        print(f"本次自定义过程组合的特征：{self.used_var}")
        print(f"本次自定义过程调整的特征：{self.adjust_var}")
        print(f"本次自定义过程调整的分箱：{self.adjust_bins}")

    def model(self):
        """
        自定义模型类
        """
        # ------------------------------------------
        # eg.
        #   from statsmodels.api import Logit
        #   return Logit
        # ------------------------------------------


if __name__ == "__main__":
    # 自定义调试用代码
    from mksc.utils import load_data, get_variable_type

    # 加载数据、变量类型划分、特征集与标签列划分
    data = load_data(mode="train")
    numeric_var, category_var, datetime_var, label_var = get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = data[label_var]
    try:
        feature[numeric_var] = feature[numeric_var].astype('float')
        feature[category_var] = feature[category_var].astype('object')
        feature[datetime_var] = feature[datetime_var].astype('datetime64')
    except Exception as e:
        print(e)
