import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def reduce_mem(df):
    """
    节约内存的一个标配函数
    """
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        assert len(col_type) == 0, "存在重名列"
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem, 100 * (start_mem-end_mem)/start_mem, (time.time()-starttime)/60))
    return df


def plot_ks(model, X_train, y_train):
    predict_train = np.array([i[1] for i in model.predict_proba(X_train)])
    fpr, tpr, thresholds = roc_curve(y_train.values, predict_train, pos_label=1)
    auc_score = auc(fpr, tpr)
    w = tpr - fpr
    ks_score = w.max()
    print(f">>> K-S 值： {ks_score}")
    ks_x = fpr[w.argmax()]
    ks_y = tpr[w.argmax()]
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='AUC=%.5f' % auc_score)
    ax.set_title('Receiver Operating Characteristic')
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    ax.plot([ks_x, ks_x], [ks_x, ks_y], '--', color='red')
    ax.text(ks_x, (ks_x + ks_y) / 2, '  KS=%.5f' % ks_score)
    ax.legend()
