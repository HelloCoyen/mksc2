from statsmodels.iolib.smpickle import load_pickle

from mksc import config
from mksc.score.card import make_card


def main():
    """
    特征评分卡制作主程序入口
    """
    odds = config.get('SCORECARD', 'ODDS')
    score = config.get('SCORECARD', 'SCORE')
    pdo = config.get('SCORECARD', 'PDO')
    feature_engineering = load_pickle("result/feature_engineering.pickle")
    woe_result = feature_engineering["woe_result"]
    model = load_pickle("result/lr.pickle")
    coefficient = list(zip(feature_engineering["feature_selected"], list(model.coef_[0])))
    coefficient.append(("intercept_", model.intercept_[0]))
    coefs = dict(coefficient)
    make_card(coefs, woe_result, odds, score, pdo)


if __name__ == "__main__":
    main()
