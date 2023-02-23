import os
import datetime
import requests
import math
import json

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('../../CheetahPy')
from cheetahpy import CheetahPy


ryan_static_metrics = {"max_hr": 191
                      ,"resting_hr": 40
                      ,'ae_threshold_hr': 148
                      ,'LTthreshold_hr': 168
                      ,'run_settings':{'cp': 356
                                      ,'w_prime': 16900
                                      ,'pmax': 642}}


class dataset(object):
    def __init__(self, ):
        self.metrics_list = ['Duration','TSS','StrydStress','Average_Heart_Rate','Max_Heartrate','Average_Power','Athlete_Weight'
                            ,'Estimated_VO2MAX','10_sec_Peak_Pace_Swim','xPace','Pace','IsoPower','Power_Index','L1_Time_in_Zone'
                            ,'L2_Time_in_Zone','L3_Time_in_Zone','L4_Time_in_Zone','L5_Time_in_Zone','L6_Time_in_Zone','L7_Time_in_Zone']
        self.metadata_list = ['VO2max_detected','Shoes','Workout_Code','Workout_Title','Indoor','Frame','Sport']
    
    def build_gc_request(self):
        base_api_endpoint = 'http://localhost:12021/Ryan%20Duecker?metrics={metrics_fields}&metadata={metadata_fields}'
        fmted_endpoint = base_api_endpoint.format(metrics_fields=','.join(self.metrics_list)
                                                ,metadata_fields=','.join(self.metadata_list))
        return fmted_endpoint
    
    def build_new_dataset(self):
        data_original = pd.read_csv(
            self.build_gc_request()
        )
        data_original.columns = [x.strip(' "') for x in data_original.columns]

        data_original['Sport'] = np.where(data_original['StrydStress']>0
                                        ,'Run'
                                        ,np.where(data_original['Average_Power']>0
                                            ,'Bike'
                                            ,np.where(data_original['10_sec_Peak_Pace_Swim']>0
                                                ,'Swim'
                                                ,'Other')))
        data_original['date'] = pd.to_datetime(data_original['date'])
        data_original['VO2max_detected'] = data_original['VO2max_detected'].astype(float)
        
        self.save_dataframe(data_original, name='gc_activitydata_ryan')

        ## Set list of activities from earlier filtered call
        self.activity_filenames = data_original[data_original['Average_Power']>0]['filename'].tolist()
    
    def calculate_activity_ef_params(self):
        ## model ef
        files_modeled = self.process_filenames()
        df = pd.DataFrame(files_modeled['modeled']
                         ,files_modeled['files']).reset_index()
        df.columns = ['files','a','b','c','rmse']

        self.save_dataframe(df, name='modeled_ef')

    def save_dataframe(self, df, name, dir='./', index_save_status=False):
        save_path = os.path.join(dir,f'{name}.csv')
        df.to_csv(save_path, index=index_save_status)
        print(f'{name} saved')

    def extract_activity_data(self, fn):
        ## Load gc api module to access individual activities 
        cp = CheetahPy()
        ac = cp.get_activity(athlete="Ryan Duecker"
                            ,activity_filename=fn)
        var_Ti = np.where(ac['temp'].mean() < -20, 20, ac['temp'].mean())
        var_HRi = ac['hr'].to_numpy()
        var_PWRi = ac['watts'].to_numpy()
        var_t = ac['secs'].to_numpy()
        cons_lag = 15

        ## Genral Formula
        # P_it = a_i + b_i*H_i,t+l + c*t*T_i
        X = np.vstack((var_HRi[cons_lag:],(var_t[:-cons_lag] * var_Ti))).T
        y = var_PWRi[:-cons_lag]
        return X, y

    def make_coef(self, X,y):
        reg = LinearRegression(fit_intercept=True).fit(X, y)
        a = reg.intercept_
        b,c = reg.coef_
        rmse = np.sqrt(((y - reg.predict(X))**2).mean())
        return a,b,c, rmse

    def process_filenames(self):
        details = {'files':self.activity_filenames
                    ,'modeled':[]}
        total_fns = len(self.activity_filenames)
        for i, fn in enumerate(self.activity_filenames):
            if i % 25 == 0:
                print("{}/{} activities modeled".format(i,total_fns))
            X, y = self.extract_activity_data(fn)
            a,b,c, rmse = self.make_coef(X,y)
            details['modeled'].append([a,b,c, rmse])
        return details

class dataset_preprocess(object):
    def __init__(self, local_activity_store=None, local_activity_model_params=None, athlete_statics=ryan_static_metrics):
        self.athlete_statics = athlete_statics
        self.local_activity_store = local_activity_store
        self.local_activity_model_params = local_activity_model_params

        if local_activity_store != None:
            self.activity_data = self.load_dataset(local_activity_store)
        if local_activity_model_params != None:
            self.modeled_data = self.load_dataset(local_activity_model_params)

    def load_dataset(self, filepath):
        data = pd.read_csv(filepath)
        return data

    def power_index_maker(self, power, duration, cp=340, w_prime=15000, pmax=448):
        theoretical_power = w_prime/duration - w_prime/(cp-pmax) + cp
        power_index = (power/theoretical_power)*100
        return power_index

    # def _calc_xpace(frame):
    #     frame['xPace'] = np.where(frame['xPace'] <= 0
    #                             ,frame['Pace']
    #                             ,frame['xPace'])
        
    def _filter_absent_data(self):
        self.activity_data = self.activity_data[~(((self.activity_data['Sport'] == 'Run') 
                                                    & (self.activity_data['Pace'] <= 0))
                                                | ((self.activity_data['Sport'] == 'Bike') & (self.activity_data['Average_Power'] <= 0))
                                                | (self.activity_data['Average_Heart_Rate'] <= 0))].copy()
        return 0

    def _reframe_data_tss(self):
        self.activity_data.rename(columns={'date':'workoutDate'}, inplace=True)
        ## transform doesn't compress the frame and instead matches index to index
        self.activity_data['day_TSS'] = self.activity_data['TSS'].groupby(self.activity_data['workoutDate']).transform('sum').fillna(0)
        return 0
    
    def _prune_relative_to_performance_metric(self, performance_lower_bound):
        # self.activity_data['performance_metric'] = np.where(self.activity_data['Duration'] < 60*60, 0, self.activity_data['performance_metric'])
        self.activity_data['performance_metric'] = np.where(self.activity_data['performance_metric'] < performance_lower_bound, 0, self.activity_data['performance_metric'])
        self.activity_data['performance_metric'] = self.activity_data['performance_metric'].replace(0,np.nan)
        self.activity_data['performance_metric'] = self.activity_data['performance_metric'].fillna(method='ffill')
        return 0
    
    def _impute_dates(self, fill_performance_forward):
        self.activity_data['date'] = self.activity_data.index
        self.activity_data['date'] = pd.to_datetime(self.activity_data['date'])
        self.activity_data = self.activity_data.sort_values(by=['date'])
        self.activity_data.index = pd.DatetimeIndex(self.activity_data['date'])
        missing_dates = pd.date_range(start=self.activity_data.index.min(), end=self.activity_data.index.max())
        self.activity_data = self.activity_data.reindex(missing_dates, fill_value=0)

        # Fill missing performance data
        if fill_performance_forward:
            self.activity_data['performance_metric'] = self.activity_data['performance_metric'].replace(0,np.nan)
            self.activity_data['performance_metric'] = self.activity_data['performance_metric'].fillna(method='ffill')
            self.activity_data = self.activity_data.dropna()
        return 0


    def pre_process(self, load_fxn, performance_fxn, performance_lower_bound=0, sport=False, fill_performance_forward=True):
        self._filter_absent_data()

        ## Use identified fxn to create performace metric for activity row
        self.activity_data['performance_metric'] = self.activity_data.apply(lambda row: performance_fxn(row, self.athlete_statics), axis=1)

        ## prune frame based of performance metric
        self._prune_relative_to_performance_metric(performance_lower_bound=performance_lower_bound)

        ## Aggregate frame to daily (+ sport data)
        agg_dict = {'day_TSS':'mean','performance_metric':'max'}
        groupby_list = ['workoutDate']
        if sport:
            groupby_list.append('Sport')
        self.activity_data = self.activity_data.groupby(groupby_list).agg(agg_dict)
        
        # Impute missing dates to create daily values + handle performance data
        self._impute_dates(fill_performance_forward=fill_performance_forward)

        return "pre-process successful"

class load_functions(dataset_preprocess):
# class load_functions(object):
    def __init__(self):
        super().__init__()
        self.metric_function_map = {
            'Daily_TSS':        self.daily_tss,
            'TIZ2_5':           self.tiz2of5,
            'TIZ1_3':           self.tiz1of3,
            'TIZ2_3':           self.tiz2of3,
            'TIZ3_3':           self.tiz3of3
        }
    
    def derive_load(self, frame: pd.DataFrame, load_metric: str) -> pd.DataFrame:
        performance_function = self.metric_function_map[load_metric]
        frame[load_metric] = performance_function(frame)
        return frame

    def daily_tss(self, frame):
        values = frame['TSS'].groupby(frame['date']).transform('sum').fillna(0)
        return values

    def tiz2of5(self, frame):
        values = frame['L2_Time_in_Zone'].groupby(frame['date']).transform('sum').fillna(0).sum(axis=1)
        return values
    
    def tiz1of3(self, frame):
        values = frame[['L1_Time_in_Zone','L2_Time_in_Zone']].groupby(frame['date']).transform('sum').fillna(0).sum(axis=1)
        return values
    
    def tiz2of3(self, frame):
        values = frame[['L3_Time_in_Zone']].groupby(frame['date']).transform('sum').fillna(0)
        return values
    
    def tiz3of3(self, frame):
        values = frame[['L4_Time_in_Zone','L5_Time_in_Zone'
                        ,'L6_Time_in_Zone','L7_Time_in_Zone']].groupby(frame['date']).transform('sum').fillna(0).sum(axis=1)
        return values

# class performance_functions(dataset_preprocess):
class performance_functions(object):
    def __init__(self, athlete_statics=ryan_static_metrics):
        # super().__init__()
        self.athlete_statics=athlete_statics
        self.metric_function_map = {
            'VO2':              self.calc_vo2,
            'Garmin VO2':       self.use_garmin_vo2,
            'AE EF':            self.calc_ae_ef,
            'Power Index':      self.use_power_index,
            'Power Index EF':   self.use_power_index_ef,
            'Mod AE Power':     self.modeled_aerobic_threshold_power
        }
    
    def derive_performance(self, frame, performance_metric: str) -> float:
        performance_function = self.metric_function_map[performance_metric]
        values = []
        for index, row in frame.iterrows():
            values.append(performance_function(row))
        frame[performance_metric] = values
        return frame

## TODO: REBUILD TO VECTOR MATH
    # def derive_performance(self, frame: pd.DataFrame, performance_metric: str) -> pd.DataFrame:
    #     performance_function = self.metric_function_map[performance_metric]
    #     frame[performance_metric] = performance_function(frame)
    #     return frame

    def calc_vo2(self, row):
        if row['Sport'] == 'Bike':
            percent_vo2 = (row['Average_Heart_Rate'] - self.athlete_statics["resting_hr"])/(self.athlete_statics["max_hr"] - self.athlete_statics["resting_hr"])
            vo2_estimated = (((row['Average_Power']/75)*1000)/row['Athlete_Weight']) / percent_vo2
        elif row['Sport'] == 'Run':
            percent_vo2 = (row['Average_Heart_Rate'] - self.athlete_statics["resting_hr"])/(self.athlete_statics["max_hr"] - self.athlete_statics["resting_hr"])
            vo2_estimated = (210/row['xPace']) / percent_vo2
        else:
            vo2_estimated =  0
        return vo2_estimated

    ## TODO: REBUILD TO VECTOR MATH
    # def calc_vo2(self, frame):
    #     percent_vo2 = (frame['Average_Heart_Rate'] - self.athlete_statics["resting_hr"])/(self.athlete_statics["max_hr"] - self.athlete_statics["resting_hr"])
    #     bike_vo2_estimated = (((frame['Average_Power']/75)*1000)/frame['Athlete_Weight']) / percent_vo2
    #     run_vo2_estimated = (210/frame['xPace']) / percent_vo2

    #     vo2_dict = {'Bike':bike_vo2_estimated
    #                ,'Run': run_vo2_estimated}
    #     for i,sport in frame['Sport'].unique():
    #         vo2_dict.update({sport:0}) if sport not in vo2_dict.keys() else 0

    #     values = frame['Sport'].apply(lambda sport: vo2_dict[sport])
    #     return values

    def use_garmin_vo2(self, row):
        vo2_estimated = 0
        if (row['Workout_Code'] != 'Rec') & (row['Sport'] in ['Run','Bike']):
            vo2_estimated = row['VO2max_detected'] # Garmin VO2 Estimation
        return vo2_estimated

    def calc_ae_ef(self, row):
        ef = 0
        if (row['Workout_Code'] == 'AE'):
            if row['Sport'] == 'Bike':
                ef = row['IsoPower']/row['Average_Heart_Rate']
            elif row['Sport'] == 'Run':
                ef = row['IsoPower']/row['Average_Heart_Rate']
        return ef

    def use_power_index(self, row):
        if row['Average_Power'] > 0:
            val = row['Power_Index']
        else:
            val = 0
        return val

    def use_power_index_ef(self, row):
        if row['Average_Power'] > 0:
            hr_range = self.athlete_statics['max_hr'] - self.athlete_statics['resting_hr']
            avg_hr_rel = row['Average_Heart_Rate'] - self.athlete_statics['resting_hr']
            relative_hr = (avg_hr_rel / hr_range)*100
            
            pi_ef = row['Power_Index']/relative_hr
            val = pi_ef
        else:
            val = 0
        return val

    def modeled_aerobic_threshold_power(self, row):
        temp = 20
        duration = 60*60
        
        if (row['a'] != 0) & (row['Duration'] > 999):
            power = row['a'] + row['b'] * self.athlete_statics['threshold_hr'] +  row['c'] * duration * temp
            return power
        else:
            return 0


if __name__ == "__main__":
    dataset_loader = dataset()
    print("Building new activity dataset")
    dataset_loader.build_new_dataset()
    print("Building new ef coef dataset")
    dataset_loader.calculate_activity_ef_params()
    print('Done!')
else:
    try:
        os.path.getsize('gc_activitydata_local.csv')
        print('Local dataset available. process using `dataset_preprocess`')
    except:
        dataset_loader = dataset()
        print('No local dataset available. build using `dataset_loader`')