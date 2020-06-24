import numpy as np
import pandas as pd
from tqdm import tqdm

from mlod import Config
from mlod.data_utils import remove_nan_values, replace_nan

import logging
logger = logging.getLogger('mlod')


class PreProcessor(object):
    def __init__(self):
        self.config = Config.load('airqo')

    def process(self, df: pd.DataFrame, test: bool, *args, **kwargs):
        raise NotImplementedError()


class MlodPreProcessor(PreProcessor):
    """
    This performs preprocessing of the data

    """
    @classmethod
    def basic_pre_process(cls, df: pd.DataFrame, test: bool=False, columns: list=None):
        """
        [in-place]Preprocesses the data as obtained from ZINDI
        
        This functions replaces all missing values (like empty strings) 
        with `numpy.nan` and converts the comma separated values into a `list` that either 
        contains `numpy.nan` or the actual number as a `float`.
        
        
        Args:
            df      :   The dataframe having the data given for the ZINDI competition
            test    :   True, if the data given is a test set, False if otherwise
                        (default: `False`)
            columns :   list of columns to handle the missing values,
                        If `None`, then all the columns will be handle for nan's
                        (default: `None`)
        
        """
        # get the column list
        if columns is None:
            columns = df.columns
        
        # replace all unfilled sections (nan) with np.nans
        for col in columns:
            df[col] = df[col].apply(lambda cell: [ replace_nan(x) for x in cell.replace('nan', ' ').split(",")])
            
        # removes the ID_*_ that is prepended in the ID defaultly
        if test:
            df['ID'] = df['ID'].map(lambda x: x.lstrip('ID_test_')).astype(str)
        else:
            df['ID'] = df['ID'].map(lambda x: x.lstrip('ID_train_')).astype(str)
    
    def expand_rows(self, df: pd.DataFrame, features_to_expand: list, id_col_name: str, repeat_other_cols: bool=True, num_steps: int = 121):
        """
        Expands the individual rows, that contain 121 values into a dataframe with row * 121 values

            Returns:
                a `pandas.DataFrame` object with the data frame with the values already expanded

        """
        # rows to distinguish the data
        row_ids = df.index

        if repeat_other_cols:
            # adds other columns to the data
            other_cols = [ feat for feat in df.columns if feat not in features_to_expand + [id_col_name] ]

        batch_rows = []
        for row_ix in tqdm(row_ids):
            row_dict = dict(zip(features_to_expand, (df.loc[row_ix, feat] for feat in features_to_expand)))

            # replicate the id
            row_dict[id_col_name] = df.loc[row_ix, id_col_name]

            # replicate the rest
            if repeat_other_cols:
                for col in other_cols:
                    row_dict[col] = [df.loc[row_ix, col]] * num_steps

            row_df = pd.DataFrame.from_dict(row_dict)
            
            batch_rows.append(row_df)
        
        # combine the data rows
        return pd.concat(batch_rows, 
                         join='inner',
                         keys=row_ids,
                         verify_integrity=True, 
                         ignore_index=True)
    
    def cyclic_encode(self, df: pd.DataFrame, columns: str):
        """
        [in-place] Encoding cyclic representations

        This is for representing features whos trend at noticed by intervals, 
        For more info: http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

        This is used in the series indices, represented by 'idx' and '*_idx' features

        """
        
        for col in columns:
            max_val = df[col].max()

            df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)

    def _add_periodic_indices(self, df: pd.DataFrame, num_days: int, num_obs: int):
        """
        Extracts the idx features, which are the indices within each row (having 121 observations)
        
        """

        # get indices of observations from each ID
        df['idx'] = df.groupby('ID').cumcount()

        # get periodic indices of observations
        obs_per_day = num_obs // num_days

        df['day_idx'] = df['idx'] // obs_per_day
        df['4day_idx'] = df['idx'] % (4*obs_per_day) #4-day series
        df['2day_idx'] = df['idx'] % (2*obs_per_day) #2-day series
        df['24hr_idx'] = df['idx'] % obs_per_day #1-day series
        df['12hr_idx'] = df['idx'] % (0.5*obs_per_day)
        df['6hr_idx'] = df['idx'] % (0.25*obs_per_day)

    @classmethod
    def calc_catersian_speed(cls, wind_dir_vals: np.ndarray, wind_speed_vals: np.ndarray):
        """
        This uses the wind direction and speed to calcuate the catersian values for the wind variable

        """

        # checks if the wind_dir values ad wind_speed_vals have the same shape
        assert wind_dir_vals.shape == wind_speed_vals.shape, 'The 2 values must be in the same dimension'

        u = - np.sin((wind_dir_vals / 360) * 2 * np.pi) * wind_speed_vals
        v = - np.cos((wind_dir_vals / 360) * 2 * np.pi) * wind_speed_vals

        return u, v

    def add_special_features(self, df: pd.DataFrame, verbose: bool = True):
        """
        [in-place] This performs the special feature engineering of the data. 
            This is specific to the AirQo challenge
        """
        # add the idx features
        self._add_periodic_indices(df, self.config.num_days, self.config.num_steps)

        # get all column with 'idx'
        # this includes:
        #  'idx', 'day_idx', '4day_idx', ...
        idx_columns = [c for c in df.columns if 'idx' in c]
        self.cyclic_encode(df, idx_columns)

        # add catersian speed features
        u, v = self.calc_catersian_speed(df['wind_dir'], df['wind_spd'])
        df['u'] = u
        df['v'] = v

        # remove the wind direction and speed
        del df['wind_dir']
        del df['wind_spd']

        # Domain specific feature engineering
        # -------------------------------------
        # Equation for the clausius-clapeyron 
        # See: https://www.theweatherprediction.com/habyhints2/646/
        
        l = 2.453e6
        rv = 461
        df['eqn'] = (l / rv) * ((1/273) - (1 / (df['temp'] + 273.15)))

    def lag_shift(self, 
                  df: pd.DataFrame, 
                  group_by_col_name: str, 
                  features: list, 
                  lag_shift_periods: list, 
                  obs_per_hour: int):

        """
        [in-place] Provides forwarded and backpassed data. This sets up
            the stage, making sequence modeling possible to machine learning models

            Given the nature of the problem, where there are `121` observation, each observation
            is collected after 1 hour. By performing lag shifts, this makes the observation for `hour x`, 
            have the reading of observation of `hour (x + 1)` [for forward pass] and for `hour (x - 1)` [for backwards pass].
            
            This is made so that machine learning models are able to see values of preceeding and successive members, and hopefully
            derive some relationship between them.

            Args:
                df (`pd.DataFrame`)             :   object that includes the columns to perform lag shifts
                group_by (`str`)                :   column to group the dataframe by. This 
                                                    will only perform lag shifts within the group
                features (`List[str]`)          :   columns whos values are passed on (shifted) amongst the rows
                lag_shift_periods (`List[int]`) :   the hourly time for which the which the contents are shifted
                obs_per_hour (`int`)            :   The number of observation per hour

        """
        for lp in tqdm(lag_shift_periods):
            for feat in features:
                forward = f'{lp}hr_fwd_lag_{feat}'
                backward = f'{lp}hr_bwd_lag_{feat}'
                forw_diff, back_diff = f'{forward}_diff', f'{backward}_diff'

                group = df.groupby(group_by_col_name)[feat]

                # Perform forward lag shift
                df[forward] = group.shift(lp * obs_per_hour)

                # Perform backwards lag shift
                df[backward] = group.shift(-lp * obs_per_hour)

                # Shift difference
                df[forw_diff] = df[feat] - df[forward]
                df[back_diff] = df[backward] - df[feat]


    def process(self, df: pd.DataFrame, test: bool=False, verbose: bool = True, cols_to_retain: list = None, force_delete_features: list = None):
        """Performs the preprocessing of the data"""
        # Working with the copy
        df = df.copy()

        # set the target name and delete the value
        init_cols = df.columns.tolist()

        if verbose: logger.info('Convert the comma-separated string into a list')
        self.basic_pre_process(df, columns=self.config.features, test=test)

        # expand the row 121
        if verbose: logger.info('Expanding the each row in the dataframe by\n'\
                      + '121 values that is in each cell')
        df = self.expand_rows(df, features_to_expand=self.config.features, id_col_name=self.config.id_col_name, repeat_other_cols=True)

        # Adding special features to the data
        if verbose: logger.info('Adding special features to data')
        self.add_special_features(df, verbose=verbose)

        # performing lag shifts to data
        if verbose: logger.info('Performing lag shifts of the features')
        self.lag_shift(df, self.config.id_col_name, self.config.eng_features, [3, 6, 9], self.config.obs_per_hour)

        # drop the unnecessary columns
        # NOTE: the expand function as addition 'ID' column, aside from the feature
        if cols_to_retain is not None:
            cols_to_delete = [ col for col in init_cols if col not in cols_to_retain and col in df.columns + [self.config.target_col_name]]

            # forcefully removes 'location' columns
            cols_to_delete.append(self.config.location_col_name)
        else:
            # removes default columns
            cols_to_delete = [self.config.id_col_name, self.config.location_col_name]
        
        # Adds features to forcefully delete
        if force_delete_features is not None:
            cols_to_delete = cols_to_delete + force_delete_features

        df.drop(cols_to_delete, axis=1, inplace=True)

        if test:
            return df
                    
        target_val = df[self.config.target_col_name]
        del df[self.config.target_col_name]

        return df, target_val


class AirQoPreProcessor(PreProcessor):
    """
    Preprocessing pipeline that pre-processes that data
        according to the notebook submitted in the discussion

        See: https://zindi.africa/competitions/airqo-ugandan-air-quality-forecast-challenge/discussions/1116
    """
    def add_statistical_features(self, df: pd.DataFrame, columns: list):
        for col in tqdm(columns):
            if col in df.columns:
                df[f'max_{col}'] = df[col].apply(np.max)
                df[f'min_{col}'] = df[col].apply(np.min)
                df[f'mean_{col}'] = df[col].apply(np.mean)
                df[f'std_{col}'] = df[col].apply(np.std)
                df[f'var_{col}'] = df[col].apply(np.var)
                df[f'median_{col}'] = df[col].apply(np.median)
                df[f'ptp_{col}'] = df[col].apply(np.ptp)

    def remove_nans(self, df: pd.DataFrame, columns: list):
        for col in tqdm(columns):
            if col in df.columns:
                df[col] = df[col].apply(remove_nan_values)

    def flatten_rows(self, df: pd.DataFrame, columns: list, num_values: int = 121):
        """
        For the columns that contains the list of 121 values, these stretches the 121 values on its own column,
            within the dataset.

            So for instance, for each 121 observations in temp.
                i.e |       temp        |
                    | 22,33,44,55,66,...|

            This adds columns associated with the values
                i.e | temp-0 | temp-1 | temp-2 | ...
                    |   22   |   33   |   44   | ...
        """ 
        # removing the data with values to flatten
        data = df[columns].copy()

        # for each values on the list, expand horizonatally
        for ix in tqdm(range(num_values)):
            for col in columns:
                df[f'{col}-{ix}'] = data[col].str[ix]

            for col in ['wind_dir', 'wind_spd']:
                # add the catersian values in the data instead of 'wind_dir' and 'wind_spd'
                u, v = MlodPreProcessor.calc_catersian_speed(df[f'wind_dir-{ix}'], df[f'wind_spd-{ix}']) 

                df[f'u-{ix}'] = u
                df[f'v-{ix}'] = v

        # delete the buffer
        del data

        # delete the wind_dir and 'wind_spd cols
        wind_dir_cols = [c for c in df.columns if 'wind_dir' in c]
        wind_spd_cols = [c for c in df.columns if 'wind_spd' in c]

        df.drop(wind_dir_cols + wind_spd_cols, inplace=True, axis=1)

    def process(self, df: pd.DataFrame, test: bool = False, verbose: bool = True, cols_to_retain: list = None, cols_to_ignore: list = ['precip']):
        """Perform the preprocessing of the data"""
        # Working with the copy
        df = df.copy()
        
        # set the target name and delete the value
        init_cols = df.columns.tolist()

        # convert the comma-separate values into lists
        MlodPreProcessor.basic_pre_process(df, test=test, columns=self.config.features)

        # flattening each row: such that each row has num_features * 121 more columns
        if verbose: logger.info('Flattening the rows')
        if verbose: logger.info("Changing 'wind_dir' and 'wind_spd', to 'u' and 'v'")
        self.flatten_rows(df, [c for c in self.config.features if c not in cols_to_ignore], self.config.num_steps)

        if verbose: logger.info('Removing the nans in rows')
        self.remove_nans(df, self.config.features)

        if verbose: logger.info('Adding Statistical Features')
        self.add_statistical_features(df, self.config.features)


        # drop the unnecessary columns
        # NOTE: the expand function as addition 'ID' column, aside from the feature
        if cols_to_retain is not None:
            cols_to_delete = [ col for col in init_cols if col not in cols_to_retain and col in list(set(df.columns + [self.config.target_col_name])) ]

            # forcefully removes 'location' columns
            cols_to_delete.append(self.config.location_col_name)
        else:
            # removes default columns
            cols_to_delete = [self.config.id_col_name, self.config.location_col_name]

        # forcefully removes the unneeded columns
        cols_to_delete = cols_to_delete + [c for c in self.config.features if c in df.columns]
        df.drop(cols_to_delete, axis=1, inplace=True)

        if test:
            return df

        target_vals = df.pop(self.config.target_col_name)
        return df, target_vals