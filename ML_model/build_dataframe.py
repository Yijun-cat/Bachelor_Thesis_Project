import os
import pandas as pd

# function to read a subject's data file into a Dataframe
# construct a dataframe with all subjects' data
def construct_df(subjects: list):
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

    # add error at t-1, t-2 as lag features
    for error in ['glideslope_error_deg', 'localizer_error_deg', 'airspeed_error_kts', 'total_error']:
        df_all[f'{error}_lag1'] = df_all.groupby(['sub_id', 'level', 'run_id'])[error].shift(1)
        df_all[f'{error}_lag2'] = df_all.groupby(['sub_id', 'level', 'run_id'])[error].shift(2)

    # dataframe for model training
    df_model = df_all.dropna().copy()
    # add a boolean column 'is_temporal_test'
    df_model['is_temporal_test'] = False
    # use last 60s as test set in within-run temporal method
    test_duration = 60
    # group by subject, level, run
    for (sub_id, level, run_id), df_run in df_model.groupby(['sub_id', 'level', 'run_id']):
        t_max = df_run['time_id'].max()
        t_split = t_max - test_duration
        id_test = df_run['time_id'] > t_split
        # reserve test set
        df_model.loc[df_run.index, 'is_temporal_test'] = id_test
    # check column names of the dataframe
    #print("===Column Names===")
    #print(list(df_model.columns))

    # feature labels
    feature_cols = ['level',
                    'HR_mean', 'HR_std', 'HR_min', 'HR_max', 'HR_range',
                    'BR_mean', 'BR_std',
                    'Rsp_amp_mean', 'Rsp_amp_std',
                    'elevation_mean',
                    'aileron_mean',
                    'gear_mean',
                    'glideslope_error_deg_lag1', 'glideslope_error_deg_lag2',
                    'localizer_error_deg_lag1', 'localizer_error_deg_lag2',
                    'airspeed_error_kts_lag1', 'airspeed_error_kts_lag2',
                    'total_error_lag1', 'total_error_lag2',
    ]
    # target labels
    target_cols = ['glideslope_error_deg', 'localizer_error_deg', 'airspeed_error_kts', 'total_error']

    # get feature and target columns
    # X = df_model[feature_cols].to_numpy()
    # y = df_model[target_cols].to_numpy()

    return df_model, feature_cols, target_cols
