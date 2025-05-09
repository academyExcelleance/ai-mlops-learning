import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from bikerent_model.config.core import config
from bikerent_model.pipeline import bikerent_pipe
from bikerent_model.processing.data_manager import load_dataset, save_pipeline

from sklearn.metrics import mean_squared_error, r2_score

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config_.training_data_file)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],  # predictors
        data[config.model_config_.target],
        test_size=config.model_config_.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config_.random_state,
    )

    # Pipeline fitting
    #print("Before pipeline X_train: ", X_train.info() )
    bikerent_pipe.fit(X_train,y_train)
    y_pred = bikerent_pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")
    # persist trained model
    save_pipeline(pipeline_to_persist= bikerent_pipe)
    # printing the score
    
if __name__ == "__main__":
    run_training()