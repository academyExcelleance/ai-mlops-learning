import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  

from bikerent_model.config.core import config
from bikerent_model.processing.features import weathersitImputer
from bikerent_model.processing.features import Mapper
from bikerent_model.processing.features import weekdayOneHotEncoder
from bikerent_model.processing.features import outlierHandler

bikerent_pipe=Pipeline([
    
    ("weathersit_imputer", weathersitImputer(variables=config.model_config_.weathersit_var)
     ),
     ##==========Mapper======##
     ("map_year", Mapper(config.model_config_.year_var)
      ),
     ("map_month", Mapper(config.model_config_.month_var)
     ),
     ("map_season", Mapper(config.model_config_.season_var)
     ),
      ("map_weathersit", Mapper(config.model_config_.weathersit_var)
     ),
      ("map_holiday", Mapper(config.model_config_.holiday_var)
     ),
      ("map_workingday", Mapper(config.model_config_.workingday_var)
     ),
      ("map_hr", Mapper(config.model_config_.hr_var)
     ),
    ('outlier_handler', outlierHandler(config.model_config_.numerical_features)),
    ('weekday_encoder', weekdayOneHotEncoder(config.model_config_.weekday_var)),
    
     ('regressor', LinearRegression())
          
     ])
