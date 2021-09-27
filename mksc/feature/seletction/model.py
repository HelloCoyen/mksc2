import pandas as pd
import statsmodels.api as sm
from numpy.linalg.linalg import LinAlgError


def stepwise_selection(x, y, threshold_in=0.01, threshold_out=0.05):
    """
    通过逐步回归筛选特征变量

    Args:
        x: 待筛选的特征数据框
        y: 数据框标签
        threshold_in: 进入模型的最大P值
        threshold_out: 退出模型的最小P值

    Returns:
        included：特征名称列表
    """
    included = []
    numeric_var = x.select_dtypes(exclude=['object', 'datetime64']).columns
    while True:
        changed = False
        # forward step
        excluded = list(set(numeric_var)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            try:
                model = sm.Logit(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            except LinAlgError:
                continue
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
 
        # backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        # use all coefs except intercept
        p_values = model.pvalues.iloc[1:]
        # null if p-values is empty
        worst_pval = p_values.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = p_values.idxmax()
            included.remove(worst_feature)
        if not changed:
            break
    return included
