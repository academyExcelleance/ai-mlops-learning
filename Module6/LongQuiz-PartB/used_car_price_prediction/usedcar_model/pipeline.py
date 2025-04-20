# pylint: disable-all

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


from sklearn.pipeline import Pipeline
import xgboost as xgb

from usedcar_model.config.core import config
from usedcar_model.processing.features import FeatureImputer, CategorialColumnsEncoder

import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
print(f"Using device: {device}")   


xgb_model = Pipeline([
    ('feature_imputer', FeatureImputer()),
    ('feature_encoder', CategorialColumnsEncoder(
        categorical_cols=config.model_config_.categorical_features
    )),
    ('regressor', xgb.XGBRegressor(
        n_estimators=config.model_config_.n_estimators,
        learning_rate=config.model_config_.learning_rate,
        max_depth=config.model_config_.max_depth,
        subsample=config.model_config_.subsample,
        colsample_bytree=config.model_config_.colsample_bytree,
        objective=config.model_config_.objective,
        random_state=config.model_config_.random_state,
        device=str(device)
    ))
])