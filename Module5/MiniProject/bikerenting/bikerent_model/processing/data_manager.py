import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from bikerent_model import __version__ as _version
from bikerent_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


##  Pre-Pipeline Preparation

# 1. Extracts the title (Mr, Ms, etc) from the name variable
def get_year_month(dataframe):
  dataframe['dteday'] = pd.to_datetime(dataframe['dteday'], format='%Y-%m-%d')
  dataframe['year'] = dataframe['dteday'].dt.year
  dataframe['month'] = dataframe['dteday'].dt.month_name()
  return dataframe
    

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self):
        pass # YOUR CODE HERE

    def fit(self, X, y=None):
        return self # YOUR CODE HERE

    def transform(self,bikeset):
       """Impute missing values in 'weekday' column."""
       imputed_bikeset = bikeset.copy()  # Create a copy of the input DataFrame
        # Find NaN entries in 'weekday' column and their row indices
       nan_indices = imputed_bikeset[imputed_bikeset['weekday'].isnull()].index

        # Extract day names from 'dteday' column for missing rows
       day_names = imputed_bikeset.loc[nan_indices, 'dteday'].dt.day_name()

        # Impute missing values with first three letters of day names
       imputed_bikeset.loc[nan_indices, 'weekday'] = day_names.str[:3]

       return imputed_bikeset


def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    bikeset = get_year_month(data_frame)       # Fetching title
    
    bikeset = WeekdayImputer().transform(bikeset)
     # drop unnecessary variables
    data_frame.drop(labels=config.model_config_.unused_fields, axis=1, inplace=True)
    #print("pre_pipeline_preparation:", data_frame.info())
    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
