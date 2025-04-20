import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from usedcar_model import __version__ as model_version
from usedcar_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {
           "id": 73015911291,
            "url": "https://wyoming.craigslist.org/ctd/d/atlanta-2019-bmw-series-430i-gran-coupe/7301591129.html",
            "region": "wyoming",
            "region_url": "https://wyoming.craigslist.org",
            "year": 2019,
            "manufacturer": "bmw",
            "model": "C4 series 430i gran coupe",
            "condition": "good",
            "cylinders": "",
            "fuel": "gas",
            "odometer": 22716,
            "title_status": "clean",
            "transmission": "other",
            "VIN": "WBA4J1C58KBM14708",
            "drive": "rwd",
            "size": "",
            "type": "coupe",
            "paint_color": "blue",
            "image_url": "https://images.craigslist.org/00Y0Y_lEUocjyRxaJz_0gw0co_600x450.jpg",
            "description": "A great car in good condition.",
            "county": "Los Angeles",
            "state": "CA",
            "lat": 34.0522,
            "long": -118.2437,
            "posting_date": "2021-04-04T03:21:07-0600"
        }
    ]
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results
