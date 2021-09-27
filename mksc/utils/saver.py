import os

from mksc import config


def save_result(data, filename, remote=False):
    """
    保存数据
    Args:
        data: DataFrame数据
        filename: 保存的文件名
        remote: 是否保存远程，默认不保存
    """
    assert filename or remote, "请至少指定一个保存模式"
    if remote:
        table_name = config.get('DATABASE', 'SCHEMA_NAME')
        engine = config.get('DATABASE', 'SAVE_ENGINE_URL')
        data.to_sql(table_name, engine, if_exists='append', index=False, chunksize=5000)
    if filename:
        data.to_csv(os.path.join(os.getcwd(), 'result', filename), index=False)
