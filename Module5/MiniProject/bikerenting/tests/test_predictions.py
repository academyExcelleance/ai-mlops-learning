"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
import pytest
import warnings

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import mean_squared_error

from bikerent_model.predict import make_prediction

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 3476
    print("************ Test Prediction*******")
    print(sample_input_data[0].head())
    # When
    result = make_prediction(input_data=sample_input_data[0])
    print(result)
    # Then
    predictions = result.get("predictions")
    print(predictions)

    print(result)

    assert isinstance(predictions, np.ndarray)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)

    print(_predictions)
    y_true = sample_input_data[1]
    print(y_true)
    
    #mse = mean_squared_error(y_test, y_pred)

    #accuracy = accuracy_score(_predictions, y_true)
    #assert accuracy > 0.8

