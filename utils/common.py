import pandas as pd

def read_data_and_split_by_ids(train_csv_path, train_ids_path, val_ids_path):
    df = pd.read_csv(train_csv_path)
    train_ids = pd.read_csv(train_ids_path)['id'].values
    val_ids = pd.read_csv(val_ids_path)['id'].values
    
    train_df = df[df['id'].isin(train_ids)]
    val_df = df[df['id'].isin(val_ids)]
    
    return train_df, val_df

def read_test_data(data_Set_name):
    if data_Set_name not in ["exit", "blueberry", "cirrhosis", "mohs", "smoker-status"]:
        raise ValueError("Invalid dataset name. Choose from: exit, blueberry, cirrhosis, mohs, smoker-status")
    
    test_csv_path = f"data/{data_Set_name}/test.csv"
    test_df = pd.read_csv(test_csv_path)
    return test_df


def verify_predictions(dataset_name):
    if dataset_name not in ["exit", "blueberry", "cirrhosis", "mohs", "smoker-status"]:
        raise ValueError("Invalid dataset name. Choose from: exit, blueberry, cirrhosis, mohs, smoker-status")
    test_csv_path = f"data/{dataset_name}/test.csv"
    test_df = pd.read_csv(test_csv_path)

    prediction_path = f"data/{dataset_name}/predictions.csv"
    predictions_df = pd.read_csv(prediction_path)
    
    if len(test_df) != len(predictions_df):
        raise ValueError(f"Length mismatch: Test data has {len(test_df)} rows, but predictions have {len(predictions_df)} rows.")
    
    if not set(test_df['id']).issubset(set(predictions_df['id'])):
        raise ValueError("Some IDs in the test data are not present in the predictions.")
    
    if not set(predictions_df['id']).issubset(set(test_df['id'])):
        raise ValueError("Some IDs in the predictions are not present in the test data.")
    
    if predictions_df.isnull().values.any():
        raise ValueError("Predictions contain missing values.")

    if predictions_df.duplicated(subset=['id']).any():
        raise ValueError("Predictions contain duplicate IDs.")
    

    return True