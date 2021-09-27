import os
import shutil
import sys

from mksc import __version__


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
    os.mkdir(os.path.join(os.getcwd(), name, r'result\model'))
    os.mkdir(os.path.join(os.getcwd(), name, 'data'))


def main():
    """
    命令行工具程序主入口
    """
    # TODO 待优化
    if len(sys.argv) == 1:
        print(f"mksc version:{__version__}")
        return "CMD FORMAT: \n\tmksc project_name1 project_name2 ...\nPlease delivery one argument at least"
    else:
        for project_name in sys.argv[1:]:
            generate_template(project_name)


if __name__ == "__main__":
    main()
