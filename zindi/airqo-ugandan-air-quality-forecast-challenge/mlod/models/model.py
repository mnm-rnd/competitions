import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from mlod import Config

import logging
logger = logging.getLogger('mlod')

class Model(object):
    def __init__(self, config_type: str, name: str):
        self.config_type = config_type
        self.config = Config.load(self.config_type)
        self.name = name
        self.model = None

        # list to store the cross-validation models
        self.cv_models = []

        # set the model properties
        self._set_model_params()

    def _set_model_params(self):
        assert self.name in self.config.available_models \
            , 'The model needs to exist in the used configuration.\n' + \
                'Available models: [{}]'.format(self.config.available_models)
        

        self.cpu_params = self.config.models[self.name]

        gpu_model_name = f'{self.name}-gpu'

        if gpu_model_name in  self.config.models:
            self.gpu_params = self.config.models[gpu_model_name]
        else:
            self.gpu_params = self.cpu_params

    @property
    def is_cv_trained(self):
        """Checks if the data is trained by cross-validation"""
        return len(self.cv_models) > 0

    def train(self, 
              x_train: pd.DataFrame, 
              y_train: pd.DataFrame,
              cv: bool=False,
              store_cv_models: bool=False,
              kfold: KFold = None,
              group: list = None, 
              use_gpu: bool=False,
              **kfold_kwargs):

        params = self.gpu_params if use_gpu else self.cpu_params
        assert self.model is None, 'The model is already trained'

        if cv:
            assert kfold is not None, "If 'cv' is True, you need to set the KFold object"
            assert 'n_splits' in kfold_kwargs, 'You need to indicate the number of n_splits'
            
            if len(self.cv_models) > 0:
                assert len(self.cv_models) != kfold_kwargs['n_splits'], 'This model was already trained on %d folds. Try changing the number of folds' % kfold_kwargs['n_splits']

            logger.info('-'*50)
            logger.info('Training in the {} folds in the data'.format(kfold_kwargs['n_splits']))
            logger.info('-'*50)

            kf = kfold(**kfold_kwargs)

            # a `dict` containing all the values
            outputs = {}

            # To store the valus of out-of-fold cross validation
            oof_vals = np.zeros(len(x_train))

            # reset the cv_models storage
            if store_cv_models: self._reset_cv_models()

            for fold, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train, group)):
                train_set = x_train.iloc[train_idx], y_train.iloc[train_idx]
                val_set = x_train.iloc[val_idx], y_train.iloc[val_idx]

                cv_model = self.cv_train(params, train_set, val_set, fold)

                if store_cv_models:
                    # adds the cross validation models to the list
                    self._add_to_cv_models(cv_model)

                # make the prediction so it's stored outside
                oof_vals[val_idx] = cv_model.predict(x_train.iloc[val_idx])

            outputs['oof'] = oof_vals
            outputs['rmse'] = self.error_eval(oof_vals, y_train)
            
            # give out the output metrics and the cross-validation models
            return outputs

        else:
            # if not training on CV

            logger.info('-'*50)
            logger.info('Training in the whole data')
            logger.info('-'*50)
            logger.info('X-shape: {} | y-shape: {}'.format(x_train, y_train))

            self.forward_train(params, x_train, y_train)

    def get_cv_models(self):
        assert self.is_cv_trained, 'Make sure you perform cross validation training first\n' \
                    + 'And set the store_cv_models=True' 

        return self.cv_models

    def _reset_cv_models(self):
        del self.cv_models
        self.cv_models = []

    def _add_to_cv_models(self, cv_model):
        # Wrap the model in the associated class
        cv_mdl = type(self)(self.config_type)
        cv_mdl.model = cv_model

        # add the model to cv models
        self.cv_models.append(cv_mdl)

    def cv_train(self, params, x_train, y_train, fold_ix):
        """Performing cross validation training"""
        raise NotImplementedError()

    def forward_train(self, params, x_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError()


    def predict(self, x_test: np.ndarray):
        assert self.model is not None, 'You need to train your model first.\n' + \
            'Make sure cv=False, when training'
        return self.model.predict(x_test)

    def error_eval(self, y_predicted, y_actual):
        raise NotImplementedError()