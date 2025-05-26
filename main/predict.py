import pandas as pd
import joblib
import argparse
import os
import glob

def load_best_model(data_name):
    models_dir = "models"
    
    model_files = glob.glob(os.path.join(models_dir, f"{data_name}_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model files found for {data_name} in {models_dir}")
    
    return joblib.load(max(model_files, key=os.path.getctime))

def make_predictions(data_name):
    data_dir = f"data/{data_name}"
    
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    sample_df = pd.read_csv(f"{data_dir}/sample_submission.csv")
    
    model = load_best_model(data_name)
    
    if hasattr(model, 'predict'):
        predictions = model.predict(test_df)
    else:
        predictions = model.predict(test_df.values)
    
    result_df = sample_df.copy()
    result_df.iloc[:, 1] = predictions
    
    output_path = f"{data_dir}/submission.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', required=True)
    args = parser.parse_args()
    
    make_predictions(args.data_name)

if __name__ == "__main__":
    main()


# python main/predict.py --data_name mohs
# python main/predict.py --data_name smoker-status