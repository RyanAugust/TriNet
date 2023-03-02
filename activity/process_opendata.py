import os
import json
import pandas as pd
import numpy as np
import datetime

class athlete(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.athlete_id = os.path.split(root_dir)[-1]
        summary_filename = '{%s}.json' % self.athlete_id
        self.athlete_summary_file = os.path.join(root_dir,summary_filename)

        self.athlete_files = next(os.walk(root_dir), (None, None, []))[2]
        [self.athlete_files.remove(file) if '.csv' not in file else None for file in self.athlete_files]
        self.athlete_files.sort()
        self.athlete_activity_filepaths = [os.path.join(root_dir,filepath) for filepath in self.athlete_files]
    
    @staticmethod
    def calc_xspeed(x, secs_id, dist_id, elev_id):
        xspeed = (
            1 + 9 *                                             # slope adj
                np.where( (x[dist_id,1:] - x[dist_id,:-1]) == 0 # safe div
                        ,0                                  
                        ,(x[elev_id,1:] - x[elev_id,:-1]) /     # delta alt
                        (x[dist_id,1:] - x[dist_id,:-1])        # delta dist
            )
        ) * (
                (x[dist_id,1:] - x[dist_id,:-1]) / # delta dist
                (x[secs_id,1:] - x[secs_id,:-1])   # delta secs
                * (60*60)                          # convert to kph
            )                                      # slope running speed
        return xspeed
    
    def get_activity_metadata(self, return_metadata=False):
        self.activity_metadata = []
        with open(os.path.join(self.root_dir,self.athlete_summary_file),'r') as f:
            ath_over = f.read()
        f.close()
        ath_over = json.loads(ath_over)
        for activity in ath_over['RIDES']:
            date_obj = datetime.datetime.strptime(activity['date'], '%Y/%m/%d %H:%M:%S %Z')
            act_metadata = {'sport':activity['sport']
                            ,'activity_dt':date_obj}
            try:
                act_metadata.update({'cp_setting':float(activity['METRICS']['cp_setting'])})
            except:
                act_metadata.update({'cp_setting':0})
            self.activity_metadata.append(act_metadata)
        if return_metadata:
            return self.activity_metadata
        else:
            return "Metadata processed"

    def extract_activity(self, activity_filepath, metrics=['secs','km','hr','power','alt'], make_xspeed=False):
        if make_xspeed and 'alt' not in metrics:
            metrics.append('alt')
        
        assert activity_filepath in self.athlete_activity_filepaths, "activity unavalable in dir"
        df = pd.read_csv(activity_filepath, engine='pyarrow')
        activity_array = df[metrics].to_numpy().T
        if make_xspeed:
            try:
                xspeed = self.calc_xspeed(activity_array, metrics.index('secs'), metrics.index('km'), metrics.index('alt'))
                xspeed = np.concatenate((np.array([0]),xspeed))
            except:
                xspeed = np.zeros(shape=activity_array.shape[1])
            activity_array = np.concatenate((activity_array,[xspeed]), axis=0)
        return activity_array

def pre_process(data_df, performance_fxn, athlete_statics, performance_lower_bound=0, sport=False):
    data_df['date'] = pd.to_datetime(data_df['date'].apply(lambda x: x.split(' ')[0]))
    data_df['xPace'] = np.where(data_df['METRICS.xPace_safe'].astype(float) <= 0, data_df['METRICS.pace'].astype(float), data_df['METRICS.xPace_safe'].astype(float))
    data_df['METRICS.average_power'] = data_df['METRICS.average_power'].apply(lambda x: float(x[0]) if type(x) == list else np.nan)
    data_df['METRICS.average_hr'] = data_df['METRICS.average_hr'].apply(lambda x: float(x[0]) if type(x) == list else np.nan)
    data_df['METRICS.athlete_weight'] = data_df['METRICS.athlete_weight'].apply(lambda x: float(x))

    data_df = data_df[~(((data_df['sport'] == 'Run') & (data_df['METRICS.pace'].astype(float) <= 0))
                | ((data_df['sport'] == 'Bike') & (data_df['METRICS.average_power'].astype(float) <= 0))
                | (data_df['METRICS.average_hr'].astype(float) <= 0))].copy()
    data_df.rename(columns={'date':'workoutDate'}, inplace=True)
    data_df['METRICS.triscore'] = data_df['METRICS.triscore'].astype(float)
    data_df['day_TSS'] = data_df['METRICS.triscore'].groupby(data_df['workoutDate']).transform('sum').fillna(0)
    data_df['performance_metric'] = data_df.apply(lambda row: performance_fxn(row, athlete_statics), axis=1)

    data_df['performance_metric'] = np.where(data_df['performance_metric'] < performance_lower_bound, 0, data_df['performance_metric'])
    data_df['performance_metric'] = data_df['performance_metric'].replace(0,np.nan)
    data_df['performance_metric'] = data_df['performance_metric'].fillna(method='ffill')

    agg_dict = {'day_TSS':'mean','performance_metric':'max'}
    if sport:
        agg_dict.update({'sport':'first'})
        data_df = data_df.sort_values('sport')
    data_df = data_df.groupby('workoutDate').agg(agg_dict)

    data_df['date'] = data_df.index
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by=['date'])
    data_df.index = pd.DatetimeIndex(data_df['date'])

    missing_dates = pd.date_range(start=data_df.index.min(), end=data_df.index.max())
    data_df = data_df.reindex(missing_dates)

    data_df['performance_metric'] = data_df['performance_metric'].replace(0,np.nan)
    # data_df['performance_metric'] = data_df['performance_metric'].fillna(method='ffill')
    data_df = data_df.dropna()
    return data_df

def calc_vo2(row, athlete_statics):
    if row['sport'] == 'Bike':
        percent_vo2 = (row['METRICS.average_hr'] - athlete_statics["resting_hr"])/(athlete_statics["max_hr"] - athlete_statics["resting_hr"])
        vo2_estimated = (((row['METRICS.average_power']/75)*1000)/row['METRICS.athlete_weight']) / percent_vo2
    elif row['sport'] == 'Run':
        percent_vo2 = (row['METRICS.average_hr'] - athlete_statics["resting_hr"])/(athlete_statics["max_hr"] - athlete_statics["resting_hr"])
        vo2_estimated = (210/row['xPace']) / percent_vo2
    else:
        vo2_estimated =  0
    return vo2_estimated

def use_garmin_vo2(row, athlete_statics):
    vo2_estimated = 0
    if (row['Workout_Code'] != 'Rec') & (row['Sport'] in ['Run','Bike']):
        vo2_estimated = row['VO2max_detected'] # Garmin VO2 Estimation
    return vo2_estimated

def calc_ae_ef(row, athlete_statics):
    ef = 0
    if (row['Workout_Code'] == 'AE'):
        if row['Sport'] == 'Bike':
            ef = row['IsoPower']/row['METRICS.average_hr']
        elif row['Sport'] == 'Run':
            ef = row['IsoPower']/row['METRICS.average_hr']
    return ef

def use_power_index(row, athlete_statics):
    if row['Average_Power'] > 0:
        val = row['Power_Index']
    else:
        val = 0
    return val

def use_power_index_ef(row, athlete_statics):
    if row['Average_Power'] > 0:
        hr_range = athlete_statics['max_hr'] - athlete_statics['resting_hr']
        avg_hr_rel = row['Average_Heart_Rate'] - athlete_statics['resting_hr']
        relative_hr = (avg_hr_rel / hr_range)*100
        
        pi_ef = row['Power_Index']/relative_hr
        val = pi_ef
    else:
        val = 0
    return val

def modeled_aerobic_threshold_power(row, athlete_statics):
    temp = 20
    duration = 60*60
    
    if (row['a'] != 0) & (row['Duration'] > 999):
        power = row['a'] + row['b']*athlete_statics['threshold_hr'] +  row['c']*duration*temp
        return power
    else:
        return 0

performance_fxns = [calc_vo2, use_garmin_vo2, calc_ae_ef, use_power_index, use_power_index_ef, modeled_aerobic_threshold_power]