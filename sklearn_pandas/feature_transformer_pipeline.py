import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional, Union, Callable


class AdvancedTransformerPipeline(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        transformers: List[tuple],
        feature_names: Optional[List[str]] = None,
        validation_mode: str = "strict",
        cache_transformations: bool = True
    ):
        self.transformers = transformers
        self.feature_names = feature_names
        self.validation_mode = validation_mode
        self.cache_transformations = cache_transformations
        self._fitted_transformers = []
        self._feature_map = {}
        
    def fit(self, X, y=None):
        # Reset fitted transformers
        self._fitted_transformers = []
        
        # Fit each transformer
        for name, transformer in self.transformers:
            if transformer is not None:
                transformer.fit(X, y)
                self._fitted_transformers.append((name, transformer))
        
        return self
    
    def transform(self, X):
        results = []
        
        for name, transformer in self._fitted_transformers:
            if transformer is not None:
                transformed = transformer.transform(X)
                results.append(transformed)
        
        if len(results) == 0:
            return X
        
        return np.hstack(results) if len(results) > 1 else results[0]
    
    def get_feature_names_out(self, input_features=None):
        feature_names = []
        
        for name, transformer in self._fitted_transformers:
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(input_features)
                feature_names.extend(names)
            else:
                feature_names.append(name)
        
        return np.array(feature_names)


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def create_feature_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    scaling_method: str = "standard"
) -> AdvancedTransformerPipeline:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    
    transformers = []
    
    # Add numeric transformers
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    if numeric_features:
        transformers.append(("numeric_scaler", scaler))
    
    # Add categorical transformers
    if categorical_features:
        transformers.append(("categorical_encoder", LabelEncoder()))
    
    return AdvancedTransformerPipeline(transformers=transformers)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns: List[str]):
        self.date_columns = date_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in self.date_columns:
            if col in X_copy.columns:
                X_copy[f'{col}_year'] = pd.to_datetime(X_copy[col]).dt.year
                X_copy[f'{col}_month'] = pd.to_datetime(X_copy[col]).dt.month
                X_copy[f'{col}_day'] = pd.to_datetime(X_copy[col]).dt.day
                X_copy[f'{col}_dayofweek'] = pd.to_datetime(X_copy[col]).dt.dayofweek
        
        return X_copy
