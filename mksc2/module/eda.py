import argparse
import pandas as pd
import pandas_profiling as pp
from mksc2.engineer.prepocess import load_data

def eda(report=False):
    """
    探索性数据分析主程序入口
    Args:
        read_local: 是否读取本地文件
        report: 是否保留分析报告，由于该报告对大数据集会占用大量时间，默认不生成

    生成以下四份文件：
        1、数据对象文件：         “data/train.pickle”
        2、变量类型配置文件：      “config/variable_type.csv”
        3、样例数据：             “data/sample.xlsx”
        4、结果报告：             “result/report.html”
    """
    # 加载数据并保存本地
    print(">>> 全量数据集加载...")
    data = load_data(mode="train")
    data.to_pickle('data/train.pickle')

    # 生成变量类别配置文件
    print(">>> 生成配置表config/variable_type.csv，请完善配置...")

    # 变量是否保留进行特征工程(isSave): 0-不保留；1-保留
    # 变量类型(Type): numeric-数值类型；category-类别类型；datetime-日期类型；label-标签列；identifier-业务标识符
    res = pd.DataFrame(zip(data.columns, [1]*len(data.columns), ['numeric']*len(data.columns), ['']*len(data.columns)),
                       columns=['Variable', 'isSave:[0/1]', 'Type:[identifier/numeric/category/datetime/label]', 'Comment'])
    res.to_csv("config/variable_type.csv", index=False)

    # 抽样探索
    print(">>> 生成探索性分析抽样data/sample.xlsx")
    sample = data.sample(min(len(data), 1000))
    sample.reset_index(drop=True, inplace=True)
    with pd.ExcelWriter('data/sample.xlsx') as writer:
        sample.to_excel(writer, sheet_name="抽样数据1000条", index=False)
        sample.corr().to_excel(writer, sheet_name="相关系数", index=False)
        sample.select_dtypes(exclude=['object', 'datetime']).describe().to_excel(writer, sheet_name="数值数据汇总")
        sample.select_dtypes(include=['object', 'datetime']).describe().to_excel(writer, sheet_name="分类数据汇总")

    # 保存分析报告
    if report and (not sample.empty):
        print(">>> 生成数据分析报告result/report.html")
        report = pp.ProfileReport(sample)
        report.to_file('result/report.html')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--report", action="store_true", help="有这个参数表示生成报告文件")
    accepted = vars(args.parse_args())
    eda(read_local=accepted.get("read_remote"), report=accepted.get("report"))
