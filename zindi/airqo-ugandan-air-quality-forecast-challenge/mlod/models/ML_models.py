import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from catboost import CatBoostRegressor

from mlod.models.model import Model

from sklearn.metrics import mean_squared_error

import logging
logger = logging.getLogger('mlod')

class LGBModel(Model):
    """Implementation of the LGBM model that was used as a base model"""
    def __init__(self, config_type: str):
        super().__init__(config_type, 'lgb')

    def cv_train(self, params, xy_train: tuple, xy_val: tuple, fold_index: int):
        train_set = lgb.Dataset(*xy_train)
        val_set = lgb.Dataset(*xy_val)

        return lgb.train(params,
                          train_set,
                          num_boost_round = 100000,
                          early_stopping_rounds = 1000, 
                          valid_sets = [train_set, val_set], 
                          verbose_eval = 2000)

    def forward_train(self, params: dict, x_train: np.ndarray, y_train: np.ndarray):
        train_set = lgb.Dataset(x_train, y_train)
        self.model = lgb.train(params, train_set, num_boost_round = 14000)

    def error_eval(self, y_predicted, y_actual):
        return np.sqrt(mean_squared_error(y_actual, y_predicted))

    @classmethod
    def from_save(cls, saved_path: str, config_type: str):
        assert Path(saved_path).exists(), 'The model path \'{}\' doesn\'t exists'

        # TODO: improve deserialization. pickle??
        c_model = cls(config_type)
        c_model.model = lgb.Booster(model_file=saved_path)

        return c_model

class CatBoostModel(Model):
    """Implementation of the CATBOOST model that was used as a meta learner"""
    def __init__(self, config_type: str):
        super().__init__(config_type, 'catboost')

    def cv_train(self, params, xy_train: tuple, xy_val: tuple, fold_index: int):
        X_train, y_train = xy_train

        model = CatBoostRegressor(eval_metric='RMSE',
                                  use_best_model=True,
                                    **params)

        model.fit(X_train, y_train,
                  eval_set=[xy_train, xy_val],
                  early_stopping_rounds=10,
                  verbose=20)

        return model

    def forward_train(self, params: dict, x_train: np.ndarray, y_train: np.ndarray):
        self.model = CatBoostRegressor(**params)
        self.model.fit(x_train, y_train)

    def error_eval(self, y_predicted, y_actual):
        return np.sqrt(mean_squared_error(y_actual, y_predicted))