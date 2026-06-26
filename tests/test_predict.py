import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from utils.predict_utils import (
    load_config,
    load_model_bundle,
    load_training_data,
    prepare_prediction_features,
)


CONFIG_PATH = 'configs/default.yaml'
MODEL_PATH = 'models/best_model_with_features.pkl'


class PredictWorkflowTest(unittest.TestCase):
    def test_prepare_prediction_features_matches_model_columns(self):
        cfg = load_config(CONFIG_PATH)
        model, feature_names = load_model_bundle(MODEL_PATH)
        training_df = load_training_data(cfg)
        input_df = training_df.drop(columns=['Churn']).head(5)

        features = prepare_prediction_features(input_df, training_df, cfg, feature_names)

        self.assertEqual(features.shape[0], 5)
        self.assertEqual(features.columns.tolist(), feature_names)
        probabilities = model.predict_proba(features)[:, 1]
        self.assertEqual(len(probabilities), 5)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_prediction_encoding_uses_training_category_mapping(self):
        cfg = load_config(CONFIG_PATH)
        _, feature_names = load_model_bundle(MODEL_PATH)
        training_df = load_training_data(cfg)
        sample_df = training_df.drop(columns=['Churn']).head(1).copy()
        sample_features = prepare_prediction_features(
            sample_df,
            training_df,
            cfg,
            feature_names,
        )

        self.assertEqual(
            int(sample_features.loc[sample_features.index[0], 'PaymentMethod']),
            2,
        )

    def test_predict_cli_writes_output_file(self):
        cfg = load_config(CONFIG_PATH)
        sample_df = load_training_data(cfg).head(3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, 'input.csv')
            output_path = os.path.join(tmp_dir, 'predictions.csv')
            sample_df.to_csv(input_path, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    'predict.py',
                    '--input',
                    input_path,
                    '--output',
                    output_path,
                    '--config',
                    CONFIG_PATH,
                    '--model-path',
                    MODEL_PATH,
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding='utf-8',
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output_df = pd.read_csv(output_path)
            self.assertEqual(len(output_df), 3)
            self.assertIn('churn_probability', output_df.columns)
            self.assertIn('predicted_churn', output_df.columns)
            self.assertTrue(output_df['churn_probability'].between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
