import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from bikerent_model.config.core import config
from bikerent_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()
        print(errors)

    return validated_data, errors


#class DataInputSchema(BaseModel):
#    PassengerId:Optional[int]
#    Pclass: Optional[int]
#    Name: Optional[str]
#    Sex: Optional[str]
#    Age: Optional[float]
#    SibSp: Optional[int]
#    Parch: Optional[int]
#    Ticket: Optional[str]
#    Fare: Optional[float]
#    Cabin: Optional[Union[str, float]]
#    Embarked: Optional[str]
    #Fare: Optional[int]

class DataInputSchema(BaseModel):
    #dteday: Optional[str]
    season: Optional[str]
    hr: Optional[str]
    holiday: Optional[str]
    weekday: Optional[str]
    workingday: Optional[str]
    weathersit: Optional[str]
    temp: Optional[float]
    atemp: Optional[float]
    hum: Optional[float]
    windspeed: Optional[float]
    #casual: Optional[float]
    #registered: Optional[float]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]