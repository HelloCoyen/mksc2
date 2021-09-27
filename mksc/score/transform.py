
def transform_score(data, score_card):
    """
    特征映射回分值
    Args:
        data: 特征表
        score_card: 评分卡

    Returns:
        返回转化后的得分
    """
    base_score = score_card[score_card['Bins'] == '-']['Score'].values[0]
    data['Score'] = base_score
    for i in range(len(data)):
        score_i = base_score
        for k in set(score_card[score_card['Bins'] != '-']['Variables']):
            bin_score = score_card[(score_card['Woe'] == data.iloc[i][k]) & (score_card['Variables'] == k)]['Score']
            score_i += bin_score.values[0]
        data['Score'].iloc[i] = score_i
    return data
