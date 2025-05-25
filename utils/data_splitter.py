import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_dataset(dataset_name, val_size=0.15):
    data_path = f"data/{dataset_name}"
    train_file = os.path.join(data_path, "train.csv")
    
    if not os.path.exists(train_file):
        print(f"No train.csv found in {data_path}")
        return
    
    df = pd.read_csv(train_file)
    print(f"\n{dataset_name.upper()}:")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if 'id' not in df.columns:
        print(f"No 'id' column found in {dataset_name}")
        return
    
    unique_ids = df['id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=val_size, random_state=42)
    
    train_df = df[df['id'].isin(train_ids)]
    val_df = df[df['id'].isin(val_ids)]
    
    print(f"Train samples: {len(train_df)} (IDs: {len(train_ids)})")
    print(f"Val samples: {len(val_df)} (IDs: {len(val_ids)})")
    
    # train_df.to_csv(os.path.join(data_path, "train_split.csv"), index=False)
    # val_df.to_csv(os.path.join(data_path, "val_split.csv"), index=False)
    
    pd.DataFrame({'id': train_ids}).to_csv(os.path.join(data_path, "train_ids.csv"), index=False)
    pd.DataFrame({'id': val_ids}).to_csv(os.path.join(data_path, "val_ids.csv"), index=False)
    
    print(f"Files saved: train_split.csv, val_split.csv, train_ids.csv, val_ids.csv")

def main():
    datasets = ['smoker-status', 'blueberry', 'cirrhosis', 'mohs']
    
    for dataset in datasets:
        split_dataset(dataset)

if __name__ == "__main__":
    main()