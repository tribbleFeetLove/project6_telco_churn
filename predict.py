"""客户流失预测命令行入口。

示例:
    python predict.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --output predictions.csv
"""

import argparse
import os
import sys

import pandas as pd

from utils.predict_utils import (
    load_config,
    load_model_bundle,
    load_training_data,
    predict_dataframe,
    prepare_prediction_features,
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='对Telco客户数据生成流失预测结果')
    parser.add_argument('--input', required=True, help='待预测CSV文件路径')
    parser.add_argument('--output', default='predictions.csv', help='预测结果CSV输出路径')
    parser.add_argument('--config', default='configs/default.yaml', help='YAML配置文件路径')
    parser.add_argument(
        '--model-path',
        default='models/best_model_with_features.pkl',
        help='已训练模型路径',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='可选概率阈值；提供后按阈值重算 predicted_churn',
    )
    return parser.parse_args()


def main() -> int:
    """执行批量预测。"""
    args = parse_args()

    if not os.path.exists(args.input):
        print(f'输入文件不存在: {args.input}', file=sys.stderr)
        return 1
    if not os.path.exists(args.model_path):
        print(f'模型文件不存在: {args.model_path}', file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    input_df = pd.read_csv(args.input)
    training_df = load_training_data(cfg)
    model, feature_names = load_model_bundle(args.model_path)

    features = prepare_prediction_features(
        input_df=input_df,
        training_df=training_df,
        cfg=cfg,
        feature_names=feature_names,
    )
    predictions = predict_dataframe(model, features)
    if args.threshold is not None:
        if not 0 <= args.threshold <= 1:
            print('threshold 必须位于 [0, 1] 区间', file=sys.stderr)
            return 1
        predictions['predicted_churn'] = (
            predictions['churn_probability'] >= args.threshold
        ).astype(int)

    output_df = input_df.copy()
    output_df['churn_probability'] = predictions['churn_probability'].values
    output_df['predicted_churn'] = predictions['predicted_churn'].values

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(args.output, index=False, encoding='utf-8-sig')

    churn_count = int(output_df['predicted_churn'].sum())
    print(f'预测完成: {len(output_df)} 条记录')
    print(f'预测流失客户: {churn_count} 条 ({churn_count / len(output_df):.2%})')
    print(f'结果已保存到: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
