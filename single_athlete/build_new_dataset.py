import pandas as pd
import requests
import numpy as np
import math
import json
import datetime

import sys
sys.path.append('../../CheetahPy')
from cheetahpy import CheetahPy
from sklearn.linear_model import LinearRegression


class dataset(object):
    def __init__(self):
        self.metrics_list = ['Duration','TSS','StrydStress','Average_Heart_Rate','Max_Heartrate','Average_Power','Athlete_Weight'
                            ,'Estimated_VO2MAX','10_sec_Peak_Pace_Swim','xPace','Pace','IsoPower','Power_Index','L1_Time_in_Zone'
                            ,'L2_Time_in_Zone','L3_Time_in_Zone','L4_Time_in_Zone','L5_Time_in_Zone','L6_Time_in_Zone','L7_Time_in_Zone']
        self.metadata_list = ['VO2max_detected','Shoes','Workout_Code','Workout_Title','Indoor','Frame','Sport']
    
    def build_gc_request(self):
        base_api_endpoint = 'http://localhost:12021/Ryan%20Duecker?metrics={metrics_fields}&metadata={metadata_fields}'
        fmted_endpoint = base_api_endpoint.format(metrics_fields=self.metrics_list
                                                ,metadata_fields=self.metadata_list)
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
        ## Load gc api module to access individual activities 
        cp = CheetahPy()
        
        ## model ef
        files_modeled = self.process_filenames()
        df = pd.DataFrame(files_modeled['modeled']
                         ,files_modeled['files']).reset_index()
        df.columns = ['files','a','b','c','rmse']

        self.save_dataframe(df, name='modeled_ef')

    def save_dataframe(df, name, dir='./', index_save_status=False):
        save_path = os.path.join([dir,f'{name}.csv'])
        df.to_csv(save_path, index=index_save_status)
        print('{name} saved')

    def extract_activity_data(self, fn):
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