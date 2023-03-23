import math
import numpy as np
import pandas as pd
import os
import glob
import yfinance as yf
from yahoofinancials import YahooFinancials
from pandas_datareader import data as pdr
import pathlib
from tqdm.auto import tqdm
from datetime import date, datetime
from scipy.signal import savgol_filter

import torch
from torch import nn as nn
from sklearn.preprocessing import StandardScaler

# TODO: Function that returns the dataset in pandas with index Date and without index Date
# TODO: Implement better the folder creation

class StockData():

    def __init__(self, 
                download=False,
                download_stock=None,
                start=None,
                end=None,
                update=False,
                update_stock=None,
                delete=False,
                delete_stock=None,
                range=0,
                folder="../../stock/"):

        self._download = download
        self._download_stock = download_stock
        self._start = start
        self._end = end
        self._update = update
        self._update_stock = update_stock
        self._delete = delete
        self._delete_stock = delete_stock
        self._range = range
        self._folder = folder

        if self._range > 0 or type(self._download_stock) is not list:
            self.stock_market = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
            self.stock_market = self.stock_market[:self._range]

        else:
            if type(self._download_stock) is list:
                self.stock_market = [i.upper() for i in self._download_stock]
            else:
                self.stock_market = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
        
        self.__main()

    def __download(self):

        if self._end == "today" or self._end is None:
            self._end = date.today().strftime("%Y-%m-%d")
        if self._start is None:
            self._start = "2017-01-01"

        yf.pdr_override()
        
        if type(self._download_stock) is not list:
            assert self._download_stock.find(",") == -1, "Wrong input format"
            if self._download_stock is not None:
                self._download_stock = self._download_stock.upper()
                self._download_stock = self._download_stock.replace('=', '')
                self._download_stock = self._download_stock.replace(' ', '')
                if self._download_stock.find('.') != -1:
                    self._download_stock = self._download_stock.replace('.', '-') 
                    
                stock_data = pdr.get_data_yahoo(self._download_stock, 
                                                start=self._start, 
                                                end=self._end,
                                                progress=True,
                                                threads=False,
                                                ignore_tz = False
                                                )
                stock_data.to_csv(self._folder + self._download_stock + ".csv", \
                                    index=True)

        else: 
            with tqdm(total=len(self.stock_market), leave=False) as bar:
                for i in self.stock_market:
                    bar.update()
                    if i.find('.') != -1:
                        i = i.replace('.', '-')
                    stock_data = pdr.get_data_yahoo(i, 
                                            start=self._start, 
                                            end=self._end,
                                            progress=False,
                                            threads=True,
                                            ignore_tz = False,
                                            )

                    stock_data.to_csv(self._folder + i + ".csv", 
                                                index=True)

    def download_stock(self):

        if type(self._download_stock) is list: 
            for i in self._download_stock:
                path = self._folder + i.upper() + '.csv'

                if os.path.isfile(path) is False or len(os.listdir(self._folder)) == 0:
                    return self.__download()
                else:
                    check = input('This file already exist, do you want download proceed [y/n]: ')
                    if check.lower() == 'y':
                        return self.__download()
                    else: raise Exception("download failed")
        else:        
            self._download_stock = self._download_stock.upper() 
            path = self._folder + self._download_stock + '.csv'

            if os.path.isfile(path) is False or len(os.listdir(self._folder)) == 0:
                return self.__download()
            else:
                check = input('This file already exist, do you want download proceed [y/n]: ')
                if check.lower() == 'y':
                    return self.__download()
                else: raise Exception("download failed")

    def __delete(self):
        files = glob.glob(self._folder + '*')
        with tqdm(total=len(files), leave=False) as bar:
            for f in files:
                os.remove(f)
                bar.update()

    def __delete_stock(self):
        self._delete_stock = self._delete_stock.upper()
        path = self._folder + self._delete_stock + ".csv"

        ## If file exists, delete it ##
        if os.path.isfile(path):
            os.remove(path)
        else:    ## Show an error ##
            print("Error: %s file not found" % path)
        
    def update(self):
        stock_market = []
        files = glob.glob(self._folder + '*')
        for f in files:
            if f.find('.') != -1:
                f = f.split(self._folder)[-1].split(".")[-2]
            f = f.replace('-', '.')
            stock_market.append(os.path.basename(f))

        with tqdm(total=len(files), leave=False) as bar:
            for file in stock_market:
                bar.update()
                self._update_stock = file
                self.__update_stock()     

    def __update_stock(self):

        self._update_stock = self._update_stock.upper()

        if self._update_stock.find('.') != -1:
            self._update_stock = self._update_stock.replace('.', '-')

        df_1 = pd.read_csv(self._folder + self._update_stock + '.csv', 
                            parse_dates=True, index_col='Date')

        if self._end == "today":
            self._end = date.today().strftime("%Y-%m-%d")
            
        if self._end is None and self._start is not None and df_1.size != 0:
            self._end = df_1.index[-1].strftime("%Y-%m-%d")
                            
        if df_1.size != 0:
            if self._start is None:
                self._start = df_1.index[0]
            else:
                self._start = self._start

            self._end = self._end
            
            yf.pdr_override()

            df = pdr.get_data_yahoo(self._update_stock, 
                                            start=self._start, 
                                            end=self._end,
                                            progress=False,
                                            threads=True,
                                            ignore_tz = True,
                                            )

            df.to_csv(self._folder + self._update_stock + '.csv', 
                            index='Date')

        if self._update is False:
            print(self._update_stock + '.csv', "Downloaded")

    def __main(self):
        path = pathlib.Path(self._folder)
        # Getting the list of directories
        if self._delete:
            self.__delete()
            print("deleted all files from ", path)
            exit()
            
        if path.exists():
            if self._download is True:
                self.__download()
        else:
            if self._download is True:
                path = self._folder 
                os.mkdir(self._folder)
                self.__download()
        
        if self._download_stock is not None:
            self.download_stock()

        if self._update is True:
            self.update()

        if self._update_stock is not None:
            self.__update_stock()

        if self._delete_stock is not None:
            self.__delete_stock()

        dir = os.listdir(path)
        if len(dir) == 0:
            var = input("The directory is empty, you want to download all stock market? [Y/n]")
            if var.lower() == 'y':
                self.__download()
            else:
                raise Exception("If you want to download one stock use --stock action")

class normalize():
    def __init__(self,
                dataframe
                ):

        self.scaler = StandardScaler()
        self.dataframe = dataframe

    def fit_transform(self):
        df_scaled = self.scaler.fit_transform(self.dataframe)
        df_scaled = pd.DataFrame(df_scaled, 
                                columns=self.dataframe.columns,
                                index=self.dataframe.index
                                )
        return df_scaled

    def inverse_transform(self):
        df_inverse = self.scaler.inverse_transform(self.dataframe)
        df_inverse = pd.DataFrame(df_inverse, 
                                columns=self.dataframe.columns,
                                index=self.dataframe.index
                                )
        return df_inverse

    def mean(self):
        df = pd.DataFrame(self.scaler.mean_.reshape(1,-1),
                            columns=self.dataframe.columns,
                            )
        return df['Close'][0]
    
    def std(self):
        df = pd.DataFrame(self.scaler.scale_.reshape(1,-1),
                            columns=self.dataframe.columns,
                            )
        return df['Close'][0]

def normalize_stock(X_train, X_test):
    target = "Close"
    tot_mean = X_train[target].mean()
    tot_std = X_train[target].std()

    for c in X_train.columns:
        mean = X_train[c].mean()
        std = X_train[c].std()

        X_train[c] = (X_train[c] - mean) / std
        X_test[c] = (X_test[c] - mean) / std

    return tot_mean, tot_std

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, 
                dataframe, 
                sequence_length=0,
                window_size = 7
                ):

        self.dataframe = dataframe

        self.sequence_length = sequence_length
        self.target= "Close"
        self.features = ['High', 'Low', 'Open', 'Volume']

        y = savgol_filter((dataframe[self.target].values), window_size, 2)
        x = (dataframe[self.features].values)

        for i in range(x.shape[1]):
            x[:, i] = savgol_filter(x[:, i], window_length=window_size, polyorder=2)
        
        self.y = torch.tensor(y).reshape(-1,1).float()
        self.X = torch.tensor(x).float()


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        if self.sequence_length == 0:
            return self.X[i], self.y[i]

        else: return x, self.y[i] 