import os
import pandas as pd

# function to read a subject's data file into a Dataframe
# construct a dataframe with all subjects' data
def construct_df(subjects: list, with_lag_feature=False):
    all_data = []
    path = os.getcwd()
    # read subjects' data file and build a Dataframe
    for sub_id in subjects:
        for level in range( 1, 5 ):
            if level == 1:
                runs = ['01', '07', '12']
            elif level == 2:
                runs = ['03', '08', '10']
            elif level == 3:
                runs = ['02', '05', '11']
            elif level == 4:
                runs = ['04', '06', '09']
            for run_id in runs:
                df = pd.read_csv(path + f"/data_ils/cp0{sub_id}/0{sub_id}_time_s/0{sub_id}_time_s_level-0{level}_run-0{run_id}_dat.csv", 
                                 index_col=False)
                df['time_id'] = df.index
                all_data.append(df)
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values(['sub_id', 'level', 'run_id'])

    # add a boolean column 'is_temporal_test'
    df_all['is_temporal_test'] = False
    df_all['is_temporal_eval'] = False
    # use last 60s as test set in within-run temporal method
    test_duration = 60
    window_length = 10
    # group by subject, level, run
    for (sub_id, level, run_id), df_run in df_all.groupby(['sub_id', 'level', 'run_id']):
        t_max = df_run['time_id'].max()
        #t_split = t_max - test_duration
        #id_test = df_run['time_id'] > t_split
        t_test_start = t_max - test_duration
        mask_test = df_run['time_id'] > t_test_start

        # evaluation times: only those test points that have a full 10 s history
        t_eval_start = t_test_start + window_length
        mask_eval = df_run['time_id'] > t_eval_start

        df_all.loc[df_run.index, "is_temporal_test"] = mask_test
        df_all.loc[df_run.index, "is_temporal_eval"] = mask_eval

    if with_lag_feature:
        # add error at t-1, t-2 as lag features
        for error in ['glideslope_error_deg', 'localizer_error_deg', 'airspeed_error_kts']:
            df_all[f'{error}_lag1'] = df_all.groupby(['sub_id', 'level', 'run_id'])[error].shift(1)
            df_all[f'{error}_lag2'] = df_all.groupby(['sub_id', 'level', 'run_id'])[error].shift(2)

        # define feature labels
        # includ lagged performance errors
        features = [
            'level',
            'HR_mean', 'HR_std', 'HR_min', 'HR_max', 'HR_range',
            'BR_mean', 'BR_std',
            'Rsp_amp_mean', 'Rsp_amp_std',
            'elevation_mean',
            'aileron_mean',
            'gear_mean',
            'glideslope_error_deg_lag1', 'glideslope_error_deg_lag2',
            'localizer_error_deg_lag1', 'localizer_error_deg_lag2',
            'airspeed_error_kts_lag1', 'airspeed_error_kts_lag2',
        ]
    else:
        # without lagged performance errors 
        features = [
            'level',
            'HR_mean', 'HR_std', 'HR_min', 'HR_max', 'HR_range',
            'BR_mean', 'BR_std',
            'Rsp_amp_mean', 'Rsp_amp_std',
            'elevation_mean',
            'aileron_mean',
            'gear_mean',
        ]

    # target labels
    targets = ['glideslope_error_deg', 'localizer_error_deg', 'airspeed_error_kts', 'total_error']
    # dataframe for model training
    df_model = df_all.dropna().copy()

    return df_model, features, targets
