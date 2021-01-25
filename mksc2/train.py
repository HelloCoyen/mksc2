import argparse
from statsmodels.iolib.smpickle import load_pickle
from mksc.utils import load_data, get_variable_type
from mksc.feature import transform
from mksc.model import training
from custom import Custom

def train(custom, **kwargs):
    """
    模型训练主程序入口
    """
    feature_engineering = load_pickle('result/feature_engineering.pickle')
    data = load_data(mode="train")
    numeric_var, category_var, datetime_var, label_var = get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = data[label_var]

    cs = Custom()
    # 自定义数据清洗
    feature, label = cs.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')

    # 自定义特征组合模块
    feature = cs.feature_combination(feature)

    # 数据处理
    feature = transform(feature, feature_engineering)
    feature = feature[feature_engineering['feature_selected']]

    # 启用自定义模型
    if custom:
        training(feature, label, use=cs.model(), **kwargs)
    else:
        training(feature, label, **kwargs)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", type=str, help="模型选择，默认自动选择最优")
    args.add_argument("--resample", action="store_true", help="有这个参数表示启用重采样")
    args.add_argument("--custom", action="store_true", help="有这个参数表示启用自定义模型")
    accepted = vars(args.parse_args())
    main(model_name=accepted['model'], resample=accepted['resample'], custom=accepted['custom'])
