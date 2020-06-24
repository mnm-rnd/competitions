"""
This contains different configurations that are used 
through out the algorithm 

"""
from mlod import SEED_NUMBER


# parameters used in the lightGBM model
_lgb_params = {
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'learning_rate': 0.08,
    'colsample_bytree': 0.85,
    'colsample_bynode': 0.85,
    'min_data_per_leaf': 25,
    'max_bin': 63,
    'num_leaves': 255,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    "metric": "rmse",
    'seed': SEED_NUMBER
}

# parameters used in the CatBoost model
_catb_params = dict(n_estimators=10000,
                    learning_rate=0.2, 
                    random_seed=SEED_NUMBER)

class AutoConfig:
    @property
    def models(self):
        raise NotImplementedError()

    @property
    def available_models(self):
        _model_name_box = []
        for name in self.models:
            if name.find('-') > -1:
                name = name.split('-')[0]

            _model_name_box.append(name)

        return _model_name_box

# ---------------------------------------------------------
# Configurations that are specific to the airqo challenge
# ---------------------------------------------------------

class AirqoConfig(AutoConfig):
    id_col_name = 'ID'
    location_col_name = 'location'
    target_col_name = 'target'

    # important features in the dataset
    features = ["temp", "precip", "rel_humidity", "wind_dir", "wind_spd", "atmos_press"]
    locations = ['A', 'B', 'C', 'D', 'E']
    num_steps = 121
    num_days = 5
    obs_per_hour = num_steps // (num_days * 24)

    # TODO: Include this to the module
    # features to involve in the feature engineering
    eng_features = ["temp", "rel_humidity", "u", "v", "atmos_press"]

    models = {
        'lgb': _lgb_params,
        'lgb-gpu': {
            'device_type': 'gpu',
            #'gpu_use_dp': 'true',
            **_lgb_params
        },        
        'catboost': _catb_params
    }

class Config:
    configs = { 'airqo': AirqoConfig }

    @classmethod
    def load(cls, option: str):
        assert option in cls.configs, f'The entered option must be among \'{cls.configs}\''
        return cls.configs[option]()