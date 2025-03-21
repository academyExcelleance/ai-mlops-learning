
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from bikerent_model.config.core import config
from bikerent_model.processing.features import weathersitImputer
from bikerent_model.processing.features import Mapper
from bikerent_model.processing.features import weekdayOneHotEncoder


def test_weathersitImputer(sample_input_data):
    
    imputer = weathersitImputer(variables=config.model_config.weathersit_var)     

    assert pd.isnull(sample_input_data[0])["weathersit"].sum() > 0

    # When
    subject =imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

    assert pd.isnull(subject["weathersit"]).sum() == 0


def test_workingday_mapper(sample_input_data):
    assert set(sample_input_data[0]["workingday"].unique()) == {"Yes", "No"}
    mapper  = Mapper(config.model_config.workingday_var)
    subject =mapper.fit(sample_input_data[0]).transform(sample_input_data[0])
    assert set(subject["workingday"].unique()) == {0, 1}


def test_weekday_encoder(sample_input_data):
  assert "weekday" in sample_input_data[0].columns, "Column 'weekday' is missing from Sample data"  
  enoder = weekdayOneHotEncoder(config.model_config.weekday_var) 
  subject =enoder.fit(sample_input_data[0]).transform(sample_input_data[0])
  assert "weekday_Mon" in subject.columns, "Column 'weekday_Mon' is missing from Sample data"  
  