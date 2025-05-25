import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_df(df, name):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    summary = {
        'dataset': name,
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'numeric_cols': len(numeric_cols),
        'categorical_cols': len(cat_cols),
        'total_nulls': df.isnull().sum().sum(),
        'null_percentage': (df.isnull().sum().sum() / df.size) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'high_cardinality_cols': [col for col in cat_cols if df[col].nunique() > len(df) * 0.8],
        'low_variance_cols': [col for col in numeric_cols if df[col].var() < 0.01],
        'skewed_cols': [col for col in numeric_cols if abs(df[col].skew()) > 2],
        'high_null_cols': [col for col in df.columns if df[col].isnull().sum() > len(df) * 0.5],
        'constant_cols': [col for col in df.columns if df[col].nunique() <= 1],
    }
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        summary['high_correlations'] = high_corr
    else:
        summary['high_correlations'] = []
    
    return summary

def create_summary(data_folder):
    datasets = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    all_summaries = []
    
    for dataset in datasets:
        dataset_path = os.path.join(data_folder, dataset)
        
        files = ['train.csv', 'test.csv', 'train_ids.csv', 'val_ids.csv']
        file_paths = {f: os.path.join(dataset_path, f) for f in files}
        
        if all(os.path.exists(path) for path in file_paths.values()):
            train_df = pd.read_csv(file_paths['train.csv'])
            test_df = pd.read_csv(file_paths['test.csv'])
            train_ids = pd.read_csv(file_paths['train_ids.csv'])['id'].values
            val_ids = pd.read_csv(file_paths['val_ids.csv'])['id'].values
            
            train_split = train_df[train_df['id'].isin(train_ids)]
            val_split = train_df[train_df['id'].isin(val_ids)]
            
            all_summaries.extend([
                analyze_df(train_split, f"{dataset}_train"),
                analyze_df(val_split, f"{dataset}_val"),
                analyze_df(test_df, f"{dataset}_test")
            ])
        else:
            print(f"Missing files in {dataset}")
    
    for dataset in datasets:
        dataset_summaries = [s for s in all_summaries if s['dataset'].startswith(dataset)]
        if dataset_summaries:
            dataset_path = os.path.join(data_folder, dataset)
            
            # Load original data for describe
            train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
            test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
            
            with open(os.path.join(dataset_path, 'summary.txt'), 'w') as f:
                f.write(f"DATASET: {dataset.upper()}\n")
                f.write("="*50 + "\n\n")
                
                f.write("BASIC INFO:\n")
                f.write(f"Train Shape: {train_df.shape}\n")
                f.write(f"Test Shape: {test_df.shape}\n")
                f.write(f"Target Column: {[col for col in train_df.columns if col not in test_df.columns]}\n")
                f.write(f"Feature Columns: {len([col for col in train_df.columns if col in test_df.columns])}\n\n")
                
                f.write("COLUMN TYPES:\n")
                f.write(str(train_df.dtypes.value_counts()) + "\n\n")
                
                f.write("MISSING VALUES:\n")
                missing = train_df.isnull().sum()[train_df.isnull().sum() > 0]
                if len(missing) > 0:
                    f.write(str(missing) + "\n\n")
                else:
                    f.write("No missing values\n\n")
                
                f.write("NUMERIC COLUMNS DESCRIBE:\n")
                numeric_cols = train_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    f.write(str(train_df[numeric_cols].describe()) + "\n\n")
                
                f.write("CATEGORICAL COLUMNS INFO:\n")
                cat_cols = train_df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    f.write(f"{col}: {train_df[col].nunique()} unique values\n")
                    f.write(f"Top values: {dict(train_df[col].value_counts().head(3))}\n")
                f.write("\n")
                
                f.write("DETAILED ANALYSIS:\n")
                for summary in dataset_summaries:
                    f.write(f"\n{summary['dataset']}:\n")
                    f.write(f"  Shape: {summary['shape']}\n")
                    f.write(f"  Memory: {summary['memory_mb']:.2f} MB\n")
                    f.write(f"  Nulls: {summary['total_nulls']} ({summary['null_percentage']:.1f}%)\n")
                    f.write(f"  Duplicates: {summary['duplicate_rows']}\n")
                    if summary['high_cardinality_cols']:
                        f.write(f"  High cardinality: {summary['high_cardinality_cols']}\n")
                    if summary['skewed_cols']:
                        f.write(f"  Skewed columns: {summary['skewed_cols']}\n")
                    if summary['high_correlations']:
                        f.write(f"  High correlations: {summary['high_correlations'][:3]}\n")
            
            print(f"Summary saved: {dataset}/summary.txt")
    
    return all_summaries

if __name__ == "__main__":
    create_summary("data")