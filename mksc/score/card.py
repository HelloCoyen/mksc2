import pickle
from math import log

import pandas as pd


def make_score(odds, score, pdo):
    """
    计算几率与得分的线性系数a , b
    Args:
        odds:  基准几率值
        score: 基准得分
        pdo: 每2倍的增量

    Returns:

    """
    # odds = P_0 / P_1
    b = float(pdo) / log(2)
    a = float(score) - b * log(float(odds))
    return a, b


def make_card(coefs, woe_result, odds, score, pdo):
    """
    建立特征维度的评分卡
    Args:
        coefs: lr特征系数
        woe_result:  特征woe值对象
        odds: 基准几率
        score: 基准几率得分
        pdo: 几率翻翻得分增量
    """
    # odds = P_0 / P_1
    a, b = make_score(odds, score, pdo)
    score_card = pd.DataFrame([['base_score', '-', '-', int(a - b * coefs['intercept_'])]],
                              columns=['Variables', 'Bins', 'Woe', 'Score'])
    for v in list(coefs.keys())[:-1]:
        if v in woe_result.keys():
            woe_result[v].insert(loc=0, column='Variables', value=v)
            woe_result[v].rename(columns={v: "Bins", 'woe_i': 'Woe'}, inplace=True)
            woe_result[v]['Score'] = woe_result[v]['Woe'].apply(lambda x: int(-x * b * coefs[v]))
            score_card = pd.concat([score_card, woe_result[v]])
        else:
            df = pd.DataFrame([[v, '-', '-', int(- b * coefs[v])]], columns=['Variables', 'Bins', 'Woe', 'Score'])
            score_card = pd.concat([score_card, df])
    score_card.to_excel('result/card.xlsx', index=False)
    with open("result/card.pickle", 'wb') as f:
        f.write(pickle.dumps(score_card))
    return score_card
