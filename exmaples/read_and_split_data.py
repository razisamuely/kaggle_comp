from utils.common import read_data_and_split_by_ids
import os 

for i in os.listdir("data/"):
    train_df, val_df = read_data_and_split_by_ids(
        f"data/{i}/train.csv",
        f"data/{i}/train_ids.csv", 
        f"data/{i}/val_ids.csv"
    )

    print(f"Dataset: ----  {i}   ----")
    print(f"Proportion of val out of total: {len(val_df) / (len(train_df) + len(val_df)):.2%}")
    print(f"Proportion of train out of total: {len(train_df) / (len(train_df) + len(val_df)):.2%}")
    print("="* 100)