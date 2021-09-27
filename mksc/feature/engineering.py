import pickle

from mksc.feature import binning
from mksc.feature import seletction
from mksc.feature import values


class FeatureEngineering(object):

    def __init__(self, feature, y, target="classify", is_bin=True, is_supervised=True, fix_missing_threshold=0.2,
                 missing_threshold=0.8, freq_threshold=0.8, unique_threshold=0.8, variance_threshold=0,
                 correlation_threshold=0.7,  **kwargs):
        self.feature = feature
        self.y = y
        self.target = target
        self.missing_threshold = missing_threshold
        self.fix_missing_threshold = fix_missing_threshold
        self.freq_threshold = freq_threshold
        self.unique_threshold = unique_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.threshold = {"missing_threshold": self.missing_threshold,
                          "freq_threshold": self.freq_threshold,
                          "unique_threshold": self.unique_threshold,
                          "correlation_threshold": self.correlation_threshold,
                          "variance_threshold": self.variance_threshold}
        self.kwargs = kwargs
        self.is_bin = is_bin
        self.is_supervised = is_supervised
        self.result = {}
        print(self.threshold)
        print(self.kwargs)

    def run(self):
        """
        特征工程过程函数,阈值参数可以自定义修改
        Returns:
            feature: 已完成特征工程的数据框
            y: 已完成特征工程的标签列
        """
        feature = self.feature.copy()
        y = self.y.copy()

        # 数据预处理
        # 缺失值处理-低于阈值的特征缺失值补充
        print(">>> 预处理: 缺失值补充")
        feature, missing_filling = values.fix_missing_value(feature, self.fix_missing_threshold)

        # 极端值处理
        print(">>> 预处理: 数值变量-极端值处理")
        feature, abnormal_value = values.fix_abnormal_value(feature, method=self.kwargs.get("method", "boundary"))

        # 标准化处理
        print(">>> 预处理: 数值变量-标准归一化")
        feature, scale_result = values.fix_scaling(feature)

        # 正态化处理
        # # 回归任务
        # if self.target == ["regression"]:
        #     standard_lambda = None
        #     if self.kwargs.get("standard", False):
        #         print(">>> 单变量预处理: BOX-COX变换")
        #         feature, standard_lambda = values.fix_standard(feature)

        # 分箱处理
        # wrapper方法+filter方法
        if self.is_bin and self.is_supervised:
            # 数值特征最优分箱，未处理的变量，暂时退出模型
            print(">>> 预处理：决策树最优分箱")
            bin_result, iv_result, woe_result = binning.tree_binning(y, feature)
            # bin_result, iv_result, woe_result = tree_binning(y, feature)
            print(f"    -- 空值规则预测：{bin_result['na_error']}")
            print(f"    -- 非空值规则预测：{bin_result['woe_adjust_result']}")
        elif self.is_bin and not self.is_supervised:
            # 等频或等距离分箱
            print(">>> 预处理：常规分箱")
            bin_result, iv_result, woe_result = binning.uniform_binning(y, feature)
        else:
            bin_result = woe_result = None
            iv_result = dict(zip(feature.columns, feature.shape[1] * [1]))

        # 特征选择
        # 单变量筛选：基于缺失率、唯一率、众数比例统计特征筛选
        print(">>> 单变量Filter: 缺失值过滤")
        missing_value = seletction.get_missing_value(feature, self.missing_threshold)
        feature.drop(missing_value['drop'], axis=1, inplace=True)
        print(f"    -- {missing_value['drop']}")

        print(">>> 单变量Filter: 类别变量-唯一率过滤")
        unique_value = seletction.get_unique_value(feature, self.unique_threshold)
        feature.drop(unique_value['drop'], axis=1, inplace=True)
        print(f"    -- {unique_value['drop']}")

        print(">>> 单变量Filter: 类别变量-众数比例过滤")
        freq_value = seletction.get_freq_value(feature, self.freq_threshold)
        feature.drop(freq_value['drop'], axis=1, inplace=True)
        print(f"    -- {freq_value['drop']}")

        print(">>> 单变量Filter: 数值变量-方差过滤")
        variance_value = seletction.get_variance_value(feature, self.variance_threshold)
        feature.drop(variance_value['drop'], axis=1, inplace=True)
        print(f"    -- {variance_value['drop']}")

        # IV筛选
        print(">>> 单变量Filter: IV值过滤")
        iv_drop = list(filter(lambda x: iv_result[x] < 0.02, iv_result))
        feature.drop(iv_drop, axis=1, inplace=True, errors="ignore")
        print(f"    -- {iv_drop}")

        # 相关性筛选
        print(">>> 多变量Filter: 相关系数过滤")
        cor_drop = seletction.get_cor_drop(feature, iv_result, self.correlation_threshold)
        feature.drop(cor_drop, axis=1, inplace=True, errors="ignore")
        print(f"    -- {cor_drop}")

        # 基础筛选
        feature_selected = list(feature.columns)

        # # woe转化
        # print(">>> 预处理: WOE转换")
        # feature = binning.woe_transform(feature, woe_result, bin_result)
        #
        # if self.target == "classify":
        # # if target == "classify":
        #     metric = "f1"
        #     estimator = DecisionTreeClassifier(min_samples_leaf=0.05, min_samples_split=0.02, max_depth=5)
        # else:
        #     metric = "r2"
        #     estimator = DecisionTreeRegressor(min_samples_leaf=0.05, min_samples_split=0.02, max_depth=5)
        #
        # selector = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5), scoring=metric)
        # # selector = SelectFromModel(estimator=estimator)
        # selector.fit(feature, y)
        # feature_selected = feature.columns[selector.get_support()]

        # from sklearn.feature_selection import SelectFromModel
        # from sklearn.linear_model import LogisticRegression
        # from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        # 带L1惩罚项的逻辑回归作为基模型的特征选择

        # 逐步回归
        # if self.kwargs.get("stepwise", False):
        #     print(">>> 多变量warpper: 逐步回归")
        #     feature_selected = seletction.model.stepwise_selection(feature, y)
        # else:
        #     # TODO 降维
        #     feature_selected = feature.columns

        # 结果保存
        self.result = {"missing_filling": missing_filling,
                       "abnormal_value": abnormal_value,
                       'scale_result': scale_result,
                       "bin_result": bin_result,
                       "iv_result": iv_result,
                       "woe_result": woe_result,
                       "missing_value": missing_value,
                       "unique_value": unique_value,
                       "freq_value": freq_value,
                       "variance_value": variance_value,
                       "iv_drop": iv_drop,
                       "cor_drop": cor_drop,
                       "feature_selected": feature_selected
                       }
        print(">>> 特征工程简单报告:")
        print(f"    缺失值过滤列数量:{missing_value['drop_number']}")
        print(f"    唯一值过滤列数量:{unique_value['drop_number']}")
        print(f"    众数值过滤列数量:{freq_value['drop_number']}")
        print(f"    方差过滤列数量:{variance_value['drop_number']}")
        print(f"    IV值过滤列数量:{len(iv_drop)}")
        print(f"    相关性过滤列数量:{len(cor_drop)}")
        print(f"    保留特征:{feature_selected}")
        print(f"    保留数量:{len(feature_selected)}")

        with open('result/feature_engineering.pickle', 'wb') as f:
            f.write(pickle.dumps(self.result))

        with open('result/short_sql.txt', 'w') as f:
            feature_selected = set([v.split("__")[0] for v in self.result["feature_selected"]])
            sql = ",".join(feature_selected)
            f.write(f"你可以使用以下取数语句避免应用集过大无法进行select *操作，注意提前剔除组合变量\n\nselect {sql} from ")
