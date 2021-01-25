import os
import configparser
import pandas as pd

def read(filename, **kwargs):
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
    elif ".ini" in filename:
        config = configparser.ConfigParser()
        config.read(filename, encoding='utf_8_sig')
        return config
    elif "select" in filename or "." not in filename:
        return pd.read_sql(filename, **kwargs)
    else:
        raise ValueError("Wrong Data Type, only [csv/pkl/xlsx/xls/txt]")

def get_config():
    return read(os.path.join(os.getcwd(), 'config', 'configuration.ini'))


config = get_config()


def check_config():
    print("\t配置文件验证中")
    return True

