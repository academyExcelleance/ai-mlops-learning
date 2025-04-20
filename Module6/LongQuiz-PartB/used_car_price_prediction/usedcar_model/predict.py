# pylint: disable-all
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from usedcar_model import __version__ as _version
from usedcar_model.config.core import config
from usedcar_model.processing.data_manager import load_pipeline
from usedcar_model.processing.data_manager import pre_pipeline_preparation
from usedcar_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
usedcar_pipe = load_pipeline(file_name = pipeline_file_name)

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = usedcar_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results

if __name__ == "__main__":
    data_in = {
            'id': [73015911291],
            'url': ['https://wyoming.craigslist.org/ctd/d/atlanta-2019-bmw-series-430i-gran-coupe/7301591129.html'],
            'region': ['wyoming'],
            'region_url': ['https://wyoming.craigslist.org'],
            'year': [2019],
            'manufacturer': ['bmw'],
            'model': ['C4 series 430i gran coupe'],
            'condition': ['good'],
            'cylinders': [''],
            'fuel': ['gas'],
            'odometer': [22716],
            'title_status': ['clean'],
            'transmission': ['other'],
            'VIN': ['WBA4J1C58KBM14708'],
            'drive': ['rwd'],
            'size': [''],
            'type': ['coupe'],
            'paint_color': ['blue'],
            'image_url': ['https://images.craigslist.org/00Y0Y_lEUocjyRxaJz_0gw0co_600x450.jpg'],
            'description': ['A great car in good condition.'],
            'county': ['Los Angeles'],
            'state': ['CA'],
            'lat': [34.0522],
            'long': [-118.2437],
            'posting_date': ['2021-04-04T03:21:07-0600']
    }
    
    make_prediction(input_data = data_in)