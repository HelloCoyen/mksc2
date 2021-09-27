import pdb

from mksc.step.eda import eda
from mksc.step.feature import feature_engineering

if __name__ == '__main__':
    eda(report=False, read_local=False)
    pdb.set_trace()
    feature_engineering(read_local=True)
    pdb.set_trace()
