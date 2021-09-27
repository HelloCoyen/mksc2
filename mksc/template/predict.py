import argparse
from datetime import date
from math import log

import pandas as pd
from statsmodels.iolib.smpickle import load_pickle

from custom import Custom
from mksc import config
from mksc.core.prepocess import load_data
from mksc.feature.binning import tree_binning
from mksc.score.card import make_score
from mksc.step.feature import processed_feature
from mksc.utils.saver import save_result


def main(mode, model_path, do_transform, read_local, woe, score=False, save_remote=False):
    # 数据、模型加载
    model, threshold = load_pickle(model_path)

    data = load_data(mode=mode, read_local=read_local)
    feature, y = processed_feature(do_transform=do_transform, mode=mode, read_local=read_local, woe=woe)

    cs = Custom()
    # 应用预测
    print(">>> 应用预测")
    res_label = pd.DataFrame(model.predict(feature), columns=['label_predict'])
    res_prob = pd.DataFrame(model.predict_proba(feature), columns=['probability_0', "probability_1"])
    res_prob['res_odds'] = res_prob['probability_0'] / res_prob["probability_1"]
    res_prob['label_threshold'] = res_prob['probability_1'].apply(lambda x: 0 if x < threshold else 1)
    res = pd.concat([data, res_label, res_prob], axis=1)

    if score:
        print(">>> 概率转换评分")
        odds = config.get('SCORECARD', 'odds')
        score = config.get('SCORECARD', 'score')
        pdo = config.get('SCORECARD', 'pdo')
        a, b = make_score(odds, score, pdo)
        res['score'] = res_prob['res_odds'].apply(lambda x: a + b * log(float(x)))
        bins = tree_binning(res[y.name], res['score'].to_frame())[0]["result"]["score"] if mode == "train" else cs.adjust_bins
        if bins:
            print(">>> 数据集分组")
            res['level'] = pd.cut(res['score'], bins)
            temp = res.groupby("level", as_index=False).count()
            temp['rate'] = temp['label_threshold'] / feature.shape[0]
            temp = temp[['level', 'rate']]
            print(temp)
            print(res.head())

    # 结果保存
    print(f">>> 结果保存中，保存模式：{save_remote}")
    res['load_date'] = str(date.today())
    save_result(res, filename=f"{mode}_result.csv", remote=save_remote)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model_path", type=str, help="模型路径")
    accepted = vars(args.parse_args())
    model_path = accepted.get('model_path')

    main(mode="predict",
         model_path=model_path,
         do_transform=True,
         read_local=False,
         score=True,
         woe=True,
         save_remote=False)
