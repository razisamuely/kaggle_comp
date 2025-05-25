import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, median_absolute_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from enum import Enum
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
import optuna
import joblib
import argparse
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)

class ProblemType(Enum):
    AUTO = 'auto'
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

class Metric(Enum):
    ACCURACY = 'accuracy'
    R2 = 'r2'
    MSE = 'mse'
    MAE = 'mae'
    MEDAE = 'medae'

def get_metric_config(metric, is_clf):
    if is_clf:
        return {'cv_scoring': 'accuracy', 'eval_func': accuracy_score, 'maximize': True}
    
    configs = {
        Metric.R2: {'cv_scoring': 'r2', 'eval_func': r2_score, 'maximize': True},
        Metric.MSE: {'cv_scoring': 'neg_mean_squared_error', 'eval_func': mean_squared_error, 'maximize': False},
        Metric.MAE: {'cv_scoring': 'neg_mean_absolute_error', 'eval_func': lambda y, p: -mean_absolute_error(y, p), 'maximize': True},
        Metric.MEDAE: {'cv_scoring': 'neg_median_absolute_error', 'eval_func': lambda y, p: -median_absolute_error(y, p), 'maximize': True}
    }
    return configs.get(metric, configs[Metric.R2])

def load_data(path):
    return pd.read_csv(path)

def preprocess(df, target_col=None, problem_type=ProblemType.AUTO):
    if target_col:
        X, y = df.drop(target_col, axis=1), df[target_col]
    else:
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', LabelEncoder(), cat_cols)
    ])
    
    if problem_type == ProblemType.AUTO:
        is_clf = y.dtype == 'object' or (y.dtype in ['int64', 'int32'] and len(y.unique()) < 20)
    else:
        is_clf = problem_type == ProblemType.CLASSIFICATION
    
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
        is_clf = True
    
    return train_test_split(X, y, test_size=0.2, random_state=42), is_clf, preprocessor

def optimize_model(trial, X_train, y_train, algo_config, metric_config):
    params = {}
    for param, config in algo_config['params'].items():
        if config['type'] == 'int':
            params[f'model__{param}'] = trial.suggest_int(param, config['low'], config['high'])
        else:
            params[f'model__{param}'] = trial.suggest_float(param, config['low'], config['high'])
    
    if TABNET_AVAILABLE and algo_config['model_class'] in [TabNetClassifier, TabNetRegressor]:
        from sklearn.base import BaseEstimator
        
        class TabNetWrapper(BaseEstimator):
            def __init__(self, model_class, **kwargs):
                self.model_class = model_class
                self.kwargs = kwargs
                
            def fit(self, X, y):
                self.model_ = self.model_class(verbose=0, seed=42, **self.kwargs)
                X_vals = X.values if hasattr(X, 'values') else X
                y_vals = y.values if hasattr(y, 'values') else y
                if len(y_vals.shape) == 1:
                    y_vals = y_vals.reshape(-1, 1)
                self.model_.fit(X_vals, y_vals)
                return self
                
            def predict(self, X):
                X_vals = X.values if hasattr(X, 'values') else X
                return self.model_.predict(X_vals)
        
        num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X_train.select_dtypes(include=['object']).columns
        tabnet_preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', LabelEncoder(), cat_cols)
        ])
        
        pipeline = Pipeline([
            ('preprocessor', tabnet_preprocessor),
            ('model', TabNetWrapper(algo_config['model_class'], **{k.replace('model__', ''): v for k, v in params.items()}))
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', algo_config['preprocessor']),
            ('model', algo_config['model_class'](random_state=42))
        ])
        pipeline.set_params(**params)
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=metric_config['cv_scoring'])
    return scores.mean()

def train_single_algo(name, algo_config, X_train, X_test, y_train, y_test, n_trials, metric_config):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: optimize_model(trial, X_train, y_train, algo_config, metric_config), n_trials=n_trials)
    
    if name == 'TABNET' and TABNET_AVAILABLE:
        # Create TabNet with preprocessing pipeline
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.compose import ColumnTransformer
        num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X_train.select_dtypes(include=['object']).columns
        tabnet_preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', LabelEncoder(), cat_cols)
        ])
        
        class TabNetWrapper:
            def __init__(self, model_class, **kwargs):
                self.model_class = model_class
                self.kwargs = kwargs
                self.preprocessor = tabnet_preprocessor
                
            def fit(self, X, y):
                X_processed = self.preprocessor.fit_transform(X)
                self.model_ = self.model_class(verbose=0, seed=42, **self.kwargs)
                y_vals = y.values if hasattr(y, 'values') else y
                if len(y_vals.shape) == 1:
                    y_vals = y_vals.reshape(-1, 1)
                self.model_.fit(X_processed, y_vals)
                return self
                
            def predict(self, X):
                X_processed = self.preprocessor.transform(X)
                return self.model_.predict(X_processed)
        
        model = TabNetWrapper(algo_config['model_class'], **study.best_params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:
        pipeline = Pipeline([
            ('preprocessor', algo_config['preprocessor']),
            ('model', algo_config['model_class'](random_state=42))
        ])
        params = {f'model__{k}': v for k, v in study.best_params.items()}
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        model = pipeline
    
    return {
        'model': model,
        'params': study.best_params,
        'score': metric_config['eval_func'](y_test, pred),
        'pred': pred
    }

def get_algos(is_clf, preprocessor):
    rf_params = {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
    }
    
    tree_params = {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0}
    }
    
    tabnet_params = {
        'n_d': {'type': 'int', 'low': 8, 'high': 64},
        'n_a': {'type': 'int', 'low': 8, 'high': 64},
        'n_steps': {'type': 'int', 'low': 3, 'high': 10},
        'gamma': {'type': 'float', 'low': 1.0, 'high': 2.0},
        'lambda_sparse': {'type': 'float', 'low': 1e-6, 'high': 1e-3}
    }
    
    algos = {}
    
    if is_clf:
        algos.update({
            'RF': {'model_class': RandomForestClassifier, 'params': rf_params, 'preprocessor': preprocessor},
            'XGB': {'model_class': xgb.XGBClassifier, 'params': tree_params, 'preprocessor': preprocessor},
            'LGBM': {'model_class': lgb.LGBMClassifier, 'params': tree_params, 'preprocessor': preprocessor}
        })
        if TABNET_AVAILABLE:
            algos['TABNET'] = {'model_class': TabNetClassifier, 'params': tabnet_params, 'preprocessor': None}
    else:
        algos.update({
            'RF': {'model_class': RandomForestRegressor, 'params': rf_params, 'preprocessor': preprocessor},
            'XGB': {'model_class': xgb.XGBRegressor, 'params': tree_params, 'preprocessor': preprocessor},
            'LGBM': {'model_class': lgb.LGBMRegressor, 'params': tree_params, 'preprocessor': preprocessor}
        })
        if TABNET_AVAILABLE:
            algos['TABNET'] = {'model_class': TabNetRegressor, 'params': tabnet_params, 'preprocessor': None}
    
    return algos

def train_all_models(X_train, X_test, y_train, y_test, is_clf, preprocessor, n_trials, metric):
    algos = get_algos(is_clf, preprocessor)
    metric_config = get_metric_config(metric, is_clf)
    results = {}
    
    for name, config in tqdm(algos.items(), desc="Training models"):
        print(f"Training {name}...")
        results[name] = train_single_algo(name, config, X_train, X_test, y_train, y_test, n_trials, metric_config)
    
    return results, metric_config

def create_report(results, y_test, is_clf, output_dir, prefix, metric, metric_config):
    best_model, best_score, best_name = None, -float('inf') if metric_config['maximize'] else float('inf'), ""
    summary_data = []
    
    for name, result in results.items():
        if is_clf:
            summary_data.append({
                'Model': name, 'Accuracy': result['score'], 'Params': str(result['params'])
            })
        else:
            mse = mean_squared_error(y_test, result['pred'])
            mae = mean_absolute_error(y_test, result['pred'])
            medae = median_absolute_error(y_test, result['pred'])
            r2 = r2_score(y_test, result['pred'])
            
            summary_data.append({
                'Model': name, f'{metric.value.upper()}': result['score'],
                'R2': r2, 'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mae, 'MedAE': medae,
                'Params': str(result['params'])
            })
        
        is_better = (result['score'] > best_score) if metric_config['maximize'] else (result['score'] < best_score)
        if is_better:
            best_score, best_model, best_name = result['score'], result['model'], name
    
    metric_name = "Accuracy" if is_clf else metric.value.upper()
    print(f"Best: {best_name} ({metric_name}: {best_score:.4f})")
    
    summary_path = os.path.join(output_dir, f'{prefix}_{best_name}_summary_{abs(best_score):.4f}.txt')
    with open(summary_path, 'w') as f:
        f.write("Model Performance Summary\n" + "="*50 + "\n\n")
        f.write(pd.DataFrame(summary_data).to_string(index=False))
        f.write(f"\n\nBest: {best_name} ({metric_name}: {best_score:.4f})")
    
    return best_model, best_name, abs(best_score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--target_col')
    parser.add_argument('--output_dir', default='models')
    parser.add_argument('--prefix', default='model')
    parser.add_argument('--metric', choices=[m.value for m in Metric], default='medae')
    parser.add_argument('--n_trials', type=int, default=1)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = load_data(args.data_path)
    (X_train, X_test, y_train, y_test), is_clf, preprocessor = preprocess(df, args.target_col)
    
    metric = Metric(args.metric)
    results, metric_config = train_all_models(X_train, X_test, y_train, y_test, is_clf, preprocessor, args.n_trials, metric)
    best_model, best_name, best_score = create_report(results, y_test, is_clf, args.output_dir, args.prefix, metric, metric_config)
    
    model_path = os.path.join(args.output_dir, f'{args.prefix}_{best_name}_{best_score:.4f}.pkl')
    joblib.dump(best_model, model_path)
    print(f"Saved: {model_path}")

if __name__ == "__main__":
    main()