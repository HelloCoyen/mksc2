import configparser
import os

__version__ = '3.0.0'


def get_config():
    cf = configparser.ConfigParser()
    filename = os.path.join(os.getcwd(), 'config', 'configuration.ini')
    try:
        assert os.path.exists(filename), f"'{filename}' doesn't exists"
        cf.read(filename, encoding='utf_8_sig')
        print("\t配置文件加载")
        return cf
    except AssertionError:
        print("\t当前工作空间为非MKSC项目目录")
        return False


def check_config(cfg):
    print("\t配置文件验证")
    if cfg:
        try:
            sql = cfg.get('DATABASE', 'TRAIN_SQL')
            filename = cfg.get('PATH', 'TRAIN_DATASET')
            engine = cfg.get('DATABASE', 'TRAIN_ENGINE_URL')
            assert sql or filename, '远程路径与本地路径至少需要提供一个用于数据集加载'
            if sql:
                assert engine, '请配置数据库链接引擎'
            print("\t配置文件验证通过")
        except AssertionError:
            print("\t配置文件验证异常")
            exit()
    else:
        print("\t当前工作空间为非MKSC项目目录，跳过文件验证")


config = get_config()
check_config(config)


