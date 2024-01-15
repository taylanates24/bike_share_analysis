import pandas as pd
import os
from tqdm import tqdm
import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, 
                        default='/home/taylan/data_analysis_project/data/pastdata', help='Dataset folder path.')
    parser.add_argument('--train_out_path', type=str, 
                        default='train_data_wo_user.csv', help='Train Dataset output path.')
    parser.add_argument('--analysis_out_path', type=str, 
                        default='analysis_data.csv', help='Analysis Dataset output path.')
    parser.add_argument('--add_user_type', type=bool,  
                        default=False, help='Analysis Dataset output path.')
    
    args = parser.parse_args()
    return args 


if __name__ == '__main__':
    
    args = arg_parser()
        
    file_paths =os.listdir(args.dataset_root_path)
    harmonized_data = []

    for file in tqdm(file_paths):
        df = pd.read_csv(os.path.join('/home/taylan/data_analysis_project/data/pastdata', file))
        df.columns = df.columns.str.lower().str.replace(" ", "")
        df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
        df['stoptime'] = pd.to_datetime(df['stoptime'], errors='coerce')
        harmonized_data.append(df)
        
    data = pd.concat(harmonized_data, ignore_index=True)

    initial_structure = data.head()

    data.dropna(subset=['starttime', 'stoptime'], inplace=True)

    data.to_csv(args.analysis_out_path)

    data['starttime'] = pd.to_datetime(data['starttime'], format='mixed')

    data['start_year_month_hour'] = data['starttime'].dt.to_period('H')
    data['day_of_week'] = data['starttime'].dt.day_of_week
    data['hour_of_day'] = data['starttime'].dt.hour
    data['month'] = data['starttime'].dt.month
    data['year'] = data['starttime'].dt.year

    net_flow = data.groupby(['start_year_month_hour', 'day_of_week', 'hour_of_day', 'month', 'year', 'startstationid']).size().reset_index(name='outflow')
    net_flow_in = data.groupby(['start_year_month_hour', 'day_of_week', 'hour_of_day', 'month', 'year', 'endstationid']).size().reset_index(name='inflow')
    net_flow_in.rename(columns={'endstationid': 'stationid'}, inplace=True)
    net_flow.rename(columns={'startstationid': 'stationid'}, inplace=True)


    merged_df = pd.merge(net_flow, net_flow_in, on=['start_year_month_hour', 'day_of_week', 'hour_of_day', 'month', 'year', 'stationid'], how='outer')

    merged_df.fillna(0, inplace=True)

    merged_df['inflow'] = merged_df['inflow'].astype(int)
    merged_df['outflow'] = merged_df['outflow'].astype(int)

    merged_df['net_flow'] = merged_df['outflow'] - merged_df['inflow']

    merged_df = merged_df.drop('inflow', axis=1)
    merged_df = merged_df.drop('outflow', axis=1)

    if args.add_user_type:
        user_data = data.groupby(['start_year_month_hour', 'usertype']).size().unstack(fill_value=0)
        merged_df = pd.merge(merged_df, user_data, on=['start_year_month_hour'], how='outer')
        
    merged_df.to_csv(args.train_out_path)
