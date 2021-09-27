# Make Scorecard(mksc)
快速构建二分类模型,标准化特征工程以及拓展制作评分卡,文件说明见

## 1. 安装工具包
```
pip install mksc
```

## 2. 创建项目
命令行工具创建项目
```
mksc project_name
```

## 3. 修改项目配置
修改`project_name\config\configuration.ini`文件，进行项目配置

主要配置数据源信息，使用URI链接远程数据库，SQL查询数据或者本地文件


## 4. 探索性数据分析与特征工程
完成上诉配置后，初始化程序`python project_name\main.py`  

先执行探索性数据分析，生成：  

* 数据报告： `project_name\result\report.html`  

* 抽样数据： `project_name\result\sample.xlsx`

* 特征配置： `project_name\config\variable_type.csv`

断点跳开后，对生成的数据描述文件进行配置，见下一步

## 5. 修改特征配置与自定义数据清洗
修改`project_name\config\variable_type.csv`文件，进行特征配置，配置列说明如下：  
* __isSave__：变量是否保留进行特征工程
    - 取值：0-不保留；1-保留
* __Type__: 变量类型
    - 取值： numeric-数值类型；category-类别类型；datetime-日期类型；label-标签列
* __Default__: 原始数据设定的默认值，该值会在后面的工程中替换成空值

完成配置后，可以选择自定义数据清洗：
        编写自定义数据清洗与特征组合过程函数`project_name\custom.py`。自定义过程封装在Custom类中，定义了2个静态方法，`clean_data`用于处理行方向的数据与值修改，`feature_combination`用于扩展列。

也可以默认数据清洗，直接回到断点键入c，开始自动化特征工程。特征工程完毕，将会生成本地文件`project_name\result\feature_engineering.pickle`

## 6. 训练模块
完成以上配置后，执行特征工程feature_engineering类，models目录下通过模型模板训练
模型结果、特征工程结果均置于`project_name\result`下.  
至此完成二分类项目构建

## 7. 评分卡与模型调整
如果训练逻辑回归模型可选制作评分卡，构建函数为mksc.score.score.main

## 8. 模型应用与预测
`python project_name\preict.py`