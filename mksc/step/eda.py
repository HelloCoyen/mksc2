import pandas as pd
import pandas_profiling as pp

from mksc.core.prepocess import load_data


def eda(report=False, read_local=False):
    """
    探索性数据分析主程序入口
    Args:
        report: 是否保留分析报告，由于该报告对大数据集会占用大量时间，默认不生成
        read_local: 是否选择读取本地文件
    生成以下四份文件：
        1、数据对象文件：         “data/train.pickle”
        2、变量类型配置文件：      “config/variable_type.csv”
        3、样例数据：             “data/sample.xlsx”
        4、结果报告：             “result/report.html”
    """
    # 加载数据并保存本地
    print(">>> 全量数据集加载")
    data = load_data(mode="train", read_local=read_local)
    # 处理同名列
    data.columns = [j + f'.{i}' if data.columns.duplicated()[i] else j for i, j in enumerate(data.columns)]
    data.to_pickle('data/train.pickle')

    # 生成变量类别配置文件
    # 变量是否保留进行特征工程(isSave): 0-不保留；1-保留
    # 变量类型(Type): numeric-数值类型；category-类别类型；datetime-日期类型；prediction-预测列；identifier-业务标识符； text-字符列
    print(">>> 生成配置表config/variable_type.csv，请完善配置")
    res = pd.DataFrame(data={'Variable': data.columns,
                             'isSave:[0/1]': [1]*len(data.columns),
                             'Type:[identifier/numeric/category/datetime/text/prediction]': ['numeric']*len(data.columns),
                             'Default': ['']*len(data.columns),
                             'Comment': ['']*len(data.columns)})
    res.to_csv("config/variable_type.csv", index=False, encoding="utf_8_sig")

    # 抽样探索
    print(">>> 生成探索性分析抽样data/sample.xlsx")
    sample = data.sample(min(len(data), 500))
    sample.reset_index(drop=True, inplace=True)
    with pd.ExcelWriter('data/sample.xlsx') as writer:
        sample.to_excel(writer, sheet_name="抽样数据500条", index=False)
        sample.corr().to_excel(writer, sheet_name="相关系数", index=False)
        sample.describe().to_excel(writer, sheet_name="数值数据汇总")
        try:
            sample.select_dtypes('object').describe().to_excel(writer, sheet_name="分类数据汇总")
        except:
            pass

    # 保存分析报告
    if report:
        assert not sample.empty, "采样集为空，请确认配置文件"
        print(">>> 生成数据分析报告result/report.html")
        report = pp.ProfileReport(sample)
        report.to_file('result/report.html')
