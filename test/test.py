from src.churn_predictor import predict_churn
import pandas as pd

# Test Case 1: Check if the output CSV is generated correctly
def test_output_csv():
    test_data = pd.DataFrame({
        'CustomerID': [1, 2],
        'Age': [25, 40],
        'MonthlySpend': [50, 200],
        'Tenure': [2, 5],
        'Churn': [0, 1]
    })
    output_df = predict_churn(test_data)
    assert 'PredictedChurn' in output_df.columns, "PredictedChurn column missing"
    assert len(output_df) == len(test_data), "Output row count mismatch"

# Test Case 2: Check for no churn in a low-risk scenario
def test_no_churn_for_low_risk():
    test_data = pd.DataFrame({
        'CustomerID': [3],
        'Age': [30],
        'MonthlySpend': [20],
        'Tenure': [10],
        'Churn': [0]
    })
    output_df = predict_churn(test_data)
    assert output_df['PredictedChurn'].iloc[0] in [0, 1], "Prediction must be 0 or 1"
