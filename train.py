"""训练入口脚本。

该脚本保留课程要求中的训练入口，实际训练流程由 main.py 统一维护。
用法:
    python train.py --config configs/default.yaml
"""

import runpy


if __name__ == '__main__':
    runpy.run_module('main', run_name='__main__')
