import pandas as pd


def read_file(filename, **kwargs):
    """
    读取数据文件
    Args:
        filename:
    Returns:
        返回数据
    """
    filename = filename.lower()
    if '.csv' in filename:
        return pd.read_csv(filename, **kwargs)
    elif '.pickle' in filename or '.pkl' in filename:
        return pd.read_pickle(filename, **kwargs)
    elif '.xls' in filename or '.xlsx' in filename:
        return pd.read_excel(filename, **kwargs)
    elif ".txt" in filename:
        return pd.read_table(filename, **kwargs)
    else:
        raise ValueError("Wrong Data Type, only [csv/pkl/xlsx/xls/txt]")