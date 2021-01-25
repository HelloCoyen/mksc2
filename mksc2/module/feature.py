import argparse
from custom import Custom
# from mksc2.feature import FeatureEngineering
from mksc2.engineer.prepocess import load_data, get_variable_type

def feature(adjust, **kwargs):
    """
    项目特征工程程序入口
    """
    # 加载数据、变量类型划分、特征集与标签列划分
    print(" >>> 数据加载...")
    data = load_data(mode="train")
    numeric_var, category_var, datetime_var, label_var = get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = data[label_var]
    print(f" >>> 当前数据规模: {data.shape}\n"
          f"     -- 数值型特征：{numeric_var}\n"
          f"     -- 类别型特征：{category_var}\n"
          f"     -- 日期型特征：{datetime_var}\n"
          f"     待筛选特征数：{feature.shape[1]}\n"
          )

    cs = Custom()
    # 自定义数据清洗
    feature, label = cs.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')

    # 自定义特征组合，全部为数值变量
    feature = cs.feature_combination(feature)

    # 调整变量，只能减少
    if adjust:
        feature = cs.feature_adjust(feature)
    cs.show()

    # 标准化特征工程
    print(">>> 启动特征工程标准进程")
    fe = FeatureEngineering(feature, label, **kwargs)
    fe.run()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--missing_threshold", type=tuple, default=(0.95, 0.05), help="缺失值阈值,默认(0.95, 0.05)")
    args.add_argument("-d", "--distinct_threshold", type=float, default=0.95, help="唯一率阈值,默认0.95")
    args.add_argument("-u", "--unique_threshold", type=float, default=0.95, help="众数阈值,默认0.95")
    args.add_argument("-a", "--abnormal_threshold", type=float, default=0.05, help="极端值阈值,默认0.05")
    args.add_argument("-c", "--correlation_threshold", type=float, default=0.7, help="相关系数阈值,默认0.7")
    args.add_argument("-v", "--variance_threshold", type=float, default=0.05, help="方差阈值,默认0.05")
    args.add_argument("--adjust", action="store_true", help="有这个参数表示启用调整的特征")
    args.add_argument("--stepwise", action="store_true", help="有这个参数表示启用逐步回归筛选")
    accepted = vars(args.parse_args())
    missing_threshold = accepted.get("missing_threshold")
    distinct_threshold = accepted.get("distinct_threshold")
    unique_threshold = accepted.get("unique_threshold")
    abnormal_threshold = accepted.get("abnormal_threshold")
    correlation_threshold = accepted.get("correlation_threshold")
    variance_threshold = accepted.get("variance_threshold")
    adjust = accepted.get("adjust")
    stepwise = accepted.get("stepwise")
    main(adjust=adjust,
         missing_threshold=missing_threshold, distinct_threshold=distinct_threshold,
         unique_threshold=unique_threshold, abnormal_threshold=abnormal_threshold,
         correlation_threshold=correlation_threshold, variance_threshold=variance_threshold,
         stepwise=stepwise)
