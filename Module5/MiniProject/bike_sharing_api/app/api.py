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
from bikerent_model import __version__ as model_version
from bikerent_model.predict import make_prediction

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
            "dteday": "2012-11-05",
            "season": "winter",
            "hr": "6am",
            "holiday": "No",
            "weekday":"Mon",
            "workingday": "Yes",
            "weathersit": "Mist",
            "temp": 6.1,
            "atemp": 3.0014000000000003,
            "hum": 49.0,
            "windspeed": 19.0012,
            "casual": 4135,
            "registered": 139
        }
    ]
}
#2012-11-05,winter,6am,No,Mon,Yes,Mist,6.1,3.0014000000000003,49.0,19.0012,4,135,139

#dteday,season,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Survival predictions with the titanic_model
    """
    print(f"Received input:  {input_data.inputs}")

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    print(f"input_df:  {input_df}")

    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    #print(f"results:  {results}")

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results

