import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='train_data_wo_user.csv', help='Dataset path.')
    
    args = parser.parse_args()
 
    return args 

class Dataset:
    
    def __init__(self, data_path, cfg, test_months=4, lags=6, rm_window=8) -> None:
        self.cfg = cfg

        self.data_path = data_path
        self.test_months = test_months
        self.lag = lags
        self.rm_window = rm_window
        
    def prepare_dataset(self):
        
        self.dataset = pd.read_csv(self.data_path)
        self.dataset = self.dataset.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        self.dataset['start_year_month_hour'] = pd.to_datetime(self.dataset['start_year_month_hour'])
        self.dataset.set_index('start_year_month_hour', inplace=True)
        self.dataset.dropna(inplace=True)
        
        split_date = self.dataset.index.max() - pd.DateOffset(months=self.test_months)

        self.train_data = self.dataset[self.dataset.index < split_date]
        self.test_data = self.dataset[self.dataset.index >= split_date]

        self.train_data = self.add_lag_and_rolling(self.train_data)
        self.test_data = self.add_lag_and_rolling(self.test_data)


        self.X_train = self.train_data.drop('net_flow', axis=1)
        self.y_train = self.train_data['net_flow']
        self.X_test = self.test_data.drop('net_flow', axis=1)
        self.y_test = self.test_data['net_flow']


    def add_lag_and_rolling(self, dataset):
        
        for lag in range(1, self.lag):
            dataset[f'lag_{lag}'] = dataset['net_flow'].shift(lag)

        dataset['rolling_mean'] = dataset['lag_1'].rolling(window=self.rm_window).mean()
        dataset['rolling_std'] = dataset['lag_1'].rolling(window=self.rm_window).std()
        dataset.dropna(inplace=True)
        
        return dataset



if __name__ == '__main__':
    
    args = arg_parser()

    datasets = Dataset(data_path=args.dataset_path, cfg=args)
    datasets.prepare_dataset()

    lgbm_model = lgb.LGBMRegressor(num_leaves=50, learning_rate=0.2, n_estimators=550)
    lgbm_model.fit(datasets.X_train, datasets.y_train)

    y_pred = lgbm_model.predict(datasets.X_test)

    mse = mean_squared_error(datasets.y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(datasets.y_test, y_pred)


    linear_model = LinearRegression()
    linear_model.fit(datasets.X_train, datasets.y_train)

    y_pred_lr = linear_model.predict(datasets.X_test)

    mse_lr = mean_squared_error(datasets.y_test, y_pred_lr)
    r2_lr = r2_score(datasets.y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.5, max_depth=5)
    xgb_model.fit(datasets.X_train, datasets.y_train)

    # Predict and evaluate the model
    y_pred_xgb = xgb_model.predict(datasets.X_test)
    mse_xgb = mean_squared_error(datasets.y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)

    r2_xgb = r2_score(datasets.y_test, y_pred_xgb)

    plt.figure(figsize=(20, 12))
    plt.scatter(range(len(datasets.y_test)), datasets.y_test, color='blue', label='Actual', alpha=0.5)
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.5)
    # plt.plot(test_dates, y_test, label='Actual', color='blue', alpha=0.3)
    # plt.plot(test_dates, y_pred, label='Predicted', color='red', alpha=0.7)

    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Net Flow')
    plt.legend()
    #plt.show()
    plt.savefig('graphs/pred_wo_user.jpg')
    print('ad')
