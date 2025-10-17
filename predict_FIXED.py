"""
Investment Prediction - Inference Script
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path


def preprocess_data(df):
    df = df.copy()

    df['offer_to_balance_ratio'] = df['offer_amount'] / (df['balance'] + 1)
    df['experienced_investor'] = (df['previous_investments'] & df['responded_before']).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100],
                             labels=['young', 'middle', 'senior', 'elderly'])
    df['balance_group'] = pd.cut(df['balance'],
                                  bins=[-np.inf, 100000, 250000, 400000, np.inf],
                                  labels=['low', 'medium', 'high', 'very_high'])
    df['offer_size'] = pd.cut(df['offer_amount'],
                               bins=[-np.inf, 40000, 60000, np.inf],
                               labels=['small', 'medium', 'large'])
    df['high_value_customer'] = ((df['balance'] > df['balance'].median()) &
                                  (df['previous_investments'] == 1)).astype(int)
    df['is_active'] = (df['responded_before'] == 1).astype(int)

    df = pd.get_dummies(df, columns=['risk_profile', 'marketing_channel', 'membership_tier',
                                     'age_group', 'balance_group', 'offer_size'],
                       drop_first=False)

    return df


def load_model_and_artifacts(model_path, feature_names_path=None):
    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    feature_names = None
    if feature_names_path and Path(feature_names_path).exists():
        print(f"Loading features: {feature_names_path}")
        feature_names = joblib.load(feature_names_path)

    return model, feature_names


def generate_predictions(test_path, model_path, output_path, feature_names_path=None):
    print("="*80)
    print("PREDICTION GENERATION")
    print("="*80)

    print(f"\nLoading test data: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test shape: {test_df.shape}")

    print("\nPreprocessing...")
    test_processed = preprocess_data(test_df)
    customer_ids = test_df['customer_id'].copy()

    X_test = test_processed.drop(['customer_id'], axis=1)

    model, feature_names = load_model_and_artifacts(model_path, feature_names_path)

    if feature_names is not None:
        print("\nAligning features...")
        missing_cols = set(feature_names) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[feature_names]
        print(f"Features: {X_test.shape[1]}")

    print("\nGenerating predictions...")
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        'customer_id': customer_ids,
        'accepted': predictions
    })

    submission.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Saved: {output_path}")
    print(f"\nDistribution:")
    print(submission['accepted'].value_counts())
    print(f"\nPositive rate: {submission['accepted'].mean():.2%}")
    print(f"\nFirst 10 rows:")
    print(submission.head(10))
    print("="*80)

    return submission


def main():
    parser = argparse.ArgumentParser(description='Investment Prediction - Generate predictions')
    parser.add_argument('--test_path', type=str, default='invest_test_public.csv',
                       help='Path to test data')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--output_path', type=str, default='submission.csv',
                       help='Path to save predictions')
    parser.add_argument('--feature_names_path', type=str, default='feature_names.pkl',
                       help='Path to feature names')

    args = parser.parse_args()

    if not Path(args.test_path).exists():
        print(f"ERROR: {args.test_path} not found!")
        sys.exit(1)

    if not Path(args.model_path).exists():
        print(f"ERROR: {args.model_path} not found!")
        sys.exit(1)

    try:
        generate_predictions(
            test_path=args.test_path,
            model_path=args.model_path,
            output_path=args.output_path,
            feature_names_path=args.feature_names_path if Path(args.feature_names_path).exists() else None
        )
        print("\nDone!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
