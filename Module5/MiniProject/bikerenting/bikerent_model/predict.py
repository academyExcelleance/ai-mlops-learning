import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikerent_model import __version__ as _version
from bikerent_model.config.core import config
from bikerent_model.pipeline import bikerent_pipe
from bikerent_model.processing.data_manager import load_pipeline
from bikerent_model.processing.data_manager import pre_pipeline_preparation
from bikerent_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
bikerent_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = bikerent_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    if not errors:

        predictions = bikerent_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        print(results)

    return results

if __name__ == "__main__":

    data_in={'dteday':['2012-12-18'],'season':['winter'],'hr':['2pm'],'holiday':['No'],'weekday':['Tue'],
                'workingday':['Yes'],'weathersit':['Clear'],'temp':[13.620000000000001],'atemp':[13.997],
                'hum':[47.0],'windspeed':[30.002599999999997],'casual':[56],'registered':[191]}
    
    make_prediction(input_data=data_in)
