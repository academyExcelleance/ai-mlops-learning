from typing import Any, List, Optional, Union

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class DataInputSchema(BaseModel):
    id: Optional[int]
    url: Optional[str]
    region: Optional[str]
    region_url: Optional[str]
    year: Optional[int]
    manufacturer: Optional[str]
    model: Optional[str]
    condition: Optional[str]
    cylinders: Optional[str]
    fuel: Optional[str]
    odometer: Optional[float]
    title_status: Optional[str]
    transmission: Optional[str]
    VIN: Optional[str]
    drive: Optional[str]
    size: Optional[str]
    type: Optional[str]
    paint_color: Optional[str]
    image_url: Optional[str]
    description: Optional[str]
    county: Optional[str]
    state: Optional[str]
    lat: Optional[float]
    long: Optional[float]
    posting_date: Optional[str]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
