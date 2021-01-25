import os
import shutil
import sys
from mksc2 import __version__
from mksc2 import Start


def generate_template(name):
    """
    创建项目的工作目录与脚本文件

    Args:
        name: 项目名
    """
    name = f'{name}_{__version__}'
    if os.path.exists(name):
        raise TypeError(f"Folder [{name}] is exists already, Please check out the work path")
    else:
        templates = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../template')
        project = os.path.join(os.getcwd(), name)
        shutil.copytree(templates, project)

    os.mkdir(os.path.join(os.getcwd(), name, 'result'))
    os.mkdir(os.path.join(os.getcwd(), name, 'data'))


def main():
    """
    命令行工具程序主入口
    """
    if len(sys.argv) == 1:
        print(f"mksc2 version:{__version__}")
        return "CMD FORMAT: \n\tmksc2 project_name1 project_name2 ...\nPlease delivery one argument at least"
    elif sys.argv[1] == 'start':
        start = Start()
        start.run()
    else:
        for project_name in sys.argv[1:]:
            generate_template(project_name)


if __name__ == "__main__":
    main()
