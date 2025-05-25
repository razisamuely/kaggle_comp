from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

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

def create_tabnet_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    return ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', LabelEncoder(), cat_cols)
    ])

def create_tabnet_pipeline(model_class, params, X_train):
    tabnet_preprocessor = create_tabnet_preprocessor(X_train)
    return Pipeline([
        ('preprocessor', tabnet_preprocessor),
        ('model', TabNetWrapper(model_class, **params))
    ])

def train_tabnet_model(model_class, params, X_train, y_train):
    tabnet_preprocessor = create_tabnet_preprocessor(X_train)
    
    class TabNetTrainer:
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
    
    model = TabNetTrainer(model_class, **params)
    model.fit(X_train, y_train)
    return model

def get_tabnet_params():
    return {
        'n_d': {'type': 'int', 'low': 8, 'high': 64},
        'n_a': {'type': 'int', 'low': 8, 'high': 64},
        'n_steps': {'type': 'int', 'low': 3, 'high': 10},
        'gamma': {'type': 'float', 'low': 1.0, 'high': 2.0},
        'lambda_sparse': {'type': 'float', 'low': 1e-6, 'high': 1e-3}
    }

def is_tabnet_model(model_class):
    return TABNET_AVAILABLE and model_class in [TabNetClassifier, TabNetRegressor]