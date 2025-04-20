from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class FeatureImputer(BaseEstimator, TransformerMixin):


    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()
        df['manufacturer'] = df['manufacturer'].fillna(df['manufacturer'].mode()[0])
        df['model'] = df['model'].fillna('unknown')
        df['odometer'] = df['odometer'].fillna(df['odometer'].mode()[0])
        df['county'] = df['county'].fillna('Unknown')
        return df
    

class CategorialColumnsEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical columns using OneHotEncoder."""

    def __init__(self, categorical_cols: List[str]):
        self.categorical_cols = categorical_cols
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.encoder.fit(X[self.categorical_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        encoded_features = self.encoder.transform(df[self.categorical_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=self.encoder.get_feature_names_out(self.categorical_cols))
        df = df.drop(columns=self.categorical_cols).reset_index(drop=True)
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        return df



