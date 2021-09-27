


# TODO
class Start(object):

    def __init__(self):
        print(">>> Welcome use mksc to build a quick model task!")
        self.accepts = {}

    def _pass(self):
        pass

    def _exit(self):
        print(">>> Thank you for use mksc, exiting...")
        exit()

    def _waiting(self):
        print('waiting until')

    def _get_param(self):
        self.accepts
        print("获取参数")

    def _interactive_input(self):
        print("交互式输入")

    def _query(self, answer, text):
        answer = answer.lower()
        if answer in ['y', 'n', 'q', 'h']:
            if answer in ["y", "n"]:
                return answer
            elif answer == "q":
                self._exit()
            elif answer == "h":
                print('\tTips:', text)
        else:
            answer = input("\tyou can type in 'h' for help or 'q' for quit\n\tA:")
            return self._query(answer, text)

    def _step(self, ask, help, true_do, false_do, **kwargs):
        answer = input(ask)
        answer = self._query(answer, help)
        if answer == 'y':
            true_do(**kwargs)
        else:
            false_do(**kwargs)
        return answer

    def run(self):

        a = "\tQ: 是否已完成配置文件configruation.ini？【y/n】\n\tA: "
        h = "\tconfigruation.ini为项目核心配置文件，项目训练前，必须完善该配置文件。"
        self._step(a, h, check_config, self._interactive_input)

        print(">>> step 1: 初始化过程-探索性数据分析过程")
        a = "\tQ: 是否执行初始化数据分析？该操作将会生成或覆盖特征配置表variable.csv【y/n】\n\tA: "
        h = "\t探索性数据分析过程将生成"
        self._step(a, h, eda, check_config)

        print(">>> step 2: 数据预处理过程")
        a = "\tQ: 是否已完成自定义清洗custom.py？完成键入Y/y, 进入等待键入N/n【y/n】\n\tA: "
        h = " custom"
        self._step(a, h, self._pass, self._waiting)

        a = "\tQ: 是否进行特征工程？\n\tA: "
        h = "feature help"
        answer_ = self._step(a, h, feature, self._pass)
        answer_ = True if answer_ == 'y' else False

        print(f">>> step {answer_ + 3}: 训练过程")
        a = "\tQ: 是否使用默认过程？【y/n】\n\tA: "
        h = "help train"
        if self._step(a, h, train, self._get_param) == "n":
            train(self.accepts)

        while True:
            a = "\tQ: 是否调整模型？【y/n】\n\tA: "
            h = "help train2"
            if self._step(a, h, self._get_param, self._exit) == "y":
                train(self.accepts)
