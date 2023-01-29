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



print('Using GC API')

data_original = pd.read_csv(
    'http://localhost:12021/Ryan%20Duecker?metrics=Duration,TSS,StrydStress,Average_Heart_Rate,Max_Heartrate,Average_Power,\
Athlete_Weight,Estimated_VO2MAX,10_sec_Peak_Pace_Swim,xPace,Pace,IsoPower,Power_Index&\
metadata=VO2max_detected,Shoes,Workout_Code,Workout_Title,Indoor,Frame,Sport'
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
data_original.to_csv('gc_activitydata_ryan.csv', index=False)

print("Modeling EF for activities")

cp = CheetahPy()
# activ = cp.get_activities("Ryan Duecker"
#                          ,start_date="2021/01/01"
#                          ,end_date="2023/02/01"
#                          ,metadata=['Workout_Code','Sport']
#                          ,activity_filenames_only=False)
fns = data_original[data_original['Average_Power']>0]['filename'].tolist()

def extract_activity_data(fn):
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

def make_coef(X,y):
    reg = LinearRegression(fit_intercept=True).fit(X, y)
    a = reg.intercept_
    b,c = reg.coef_
    rmse = np.sqrt(((y - reg.predict(X))**2).mean())
    return a,b,c, rmse

def process_filenames(fns):
    details = {'files':fns
                ,'modeled':[]}
    total_fns = len(fns)
    for i, fn in enumerate(fns):
        if i % 25 == 0:
            print("{}/{} activities modeled".format(i,total_fns))
        X, y = extract_activity_data(fn)
        a,b,c, rmse = make_coef(X,y)
        details['modeled'].append([a,b,c, rmse])
    return details

files_modeled = process_filenames(fns)
df = pd.DataFrame(files_modeled['modeled'],files_modeled['files']).reset_index()
df.columns = ['files','a','b','c','rmse']
df.to_csv('modeled_ef.csv')