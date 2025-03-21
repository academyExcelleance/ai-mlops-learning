from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class weathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self,variables: str):
         self.most_frequent_category = None
         if not isinstance(variables, str):
            raise ValueError("variables should be a str")

         self.variables = variables

         
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.most_frequent_category=X[self.variables].mode()[0]
        return self 
    

    def transform(self,X):
        X_copy = X.copy()  # Create a copy to avoid modifying the original DataFrame
        X_copy[self.variables]=X[self.variables].fillna(self.most_frequent_category)
        return X_copy


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, features_to_map):
        self.features_to_map = features_to_map
        self.mapping_dicts = {}
      

    def fit(self, X, y=None):
        """
        Fit the mapper by creating mapping dictionaries for each feature.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y: Ignored.

        Returns:
            self: The fitted mapper.
        """
        #print(X.info())
        for feature in [self.features_to_map]:
            # Assuming ordinal relationship for mapping
           # print("Features========",feature)
            #unique_values = sorted(X[feature].dropna().astype(str).unique())
            unique_values = sorted(X[feature].unique())
            #print("Unique Features========",unique_values)
            self.mapping_dicts[feature] = {val: i for i, val in enumerate(unique_values)}
        return self

    def transform(self, X):
        """
        Transform the data by applying the mappings.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with mapped values.
        """
        X_copy = X.copy()
        #print("Before Mapper:",  X_copy.info())
        for feature, mapping_dict in self.mapping_dicts.items():
            #print("Info:", X_copy[feature].unique())
            #print("******************** Feature:************************", feature)
            #print("******************** mapping_dict:************************", mapping_dict)
            X_copy[feature] = X_copy[feature].map(mapping_dict)           
        return X_copy
# class Mapper(BaseEstimator, TransformerMixin):
#     """Categorical variable mapper."""

#     def __init__(self, variables: str, mappings: dict):

#         if not isinstance(variables, str):
#             raise ValueError("variables should be a str")

#         self.variables = variables
#         self.mappings = mappings

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         # we need the fit statement to accomodate the sklearn pipeline
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         X[self.variables] = X[self.variables].map(self.mappings).astype(int)

#         return X

class outlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, features):
        self.features = features
        self.bounds = {}  # Store upper and lower bounds for each feature

    def fit(self, X, y=None):
        """
        Calculate and store the upper and lower bounds for each feature.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y: Ignored.

        Returns:
            self: The fitted OutlierHandler.
        """
        for feature in self.features:
            Q1 = X[feature].quantile(0.25)
            Q3 = X[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.bounds[feature] = (lower_bound, upper_bound)
        return self

    def transform(self, X):
        """
        Transform the data by changing outlier values to bounds.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with outlier values capped.
        """
        X_copy = X.copy()  # Create a copy to avoid modifying the original DataFrame
        for feature, (lower_bound, upper_bound) in self.bounds.items():
            X_copy[feature] = np.clip(X_copy[feature], lower_bound, upper_bound)
        return X_copy
        

class weekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variables:str,handle_unknown='ignore'):
        self.handle_unknown = handle_unknown
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown=self.handle_unknown)  
        self.weekday_categories = None  # To store the categories during fit
        self.variables = variables
    # def fit(self, X, y=None):
    #     self.weekday_categories = X['weekday'].unique()  # Get unique weekday values
    #     self.ohe.fit(X[['weekday']])  # Fit the OneHotEncoder on the 'weekday' column
    #     return self
    def fit(self, X, y=None):
        self.weekday_categories = X[self.variables].unique()  # Get unique weekday values
        self.ohe.fit(X[[self.variables]])
        #self.ohe.fit(X[self.variables].values.reshape(-1,1))  # Fit the OneHotEncoder on the 'weekday' column
        return self

    def transform(self, X):
        # Ensure the input DataFrame has the 'weekday' column:
        if 'weekday' not in X.columns:
            raise ValueError("Input DataFrame must contain the 'weekday' column.")
        
        #print("self.variables",self.variables)
        #print("X[self.variables].shape",X[self.variables].shape)
        
        # One-hot encode the 'weekday' column:
        encoded_weekday = self.ohe.transform(X[[self.variables]])
        #print("After transform:", encoded_weekday)
        # Create column names for the encoded features:
        encoded_feature_names = [f"weekday_{cat}" for cat in self.ohe.categories_[0]]
        #print("encoded_feature_names:", encoded_feature_names)

        # Create a DataFrame with the encoded features:
        encoded_weekday_df = pd.DataFrame(encoded_weekday, columns=encoded_feature_names, index=X.index)
        #print("encoded_weekday_df:", encoded_weekday_df.info())
        #print("X:", X.info())

        # Concatenate the encoded features with the original DataFrame:
        X_encoded = pd.concat([X, encoded_weekday_df], axis=1)
        #print("X_encoded1", X_encoded.info())
        # Drop the original 'weekday' column:
        X_encoded = X_encoded.drop(columns=[self.variables])
        #print("X_encoded2:", X_encoded.info())

        return X_encoded