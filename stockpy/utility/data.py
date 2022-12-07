import pandas as pd
import yfinance as yf
import argparse
import os
import glob
from yahoofinancials import YahooFinancials
from pandas_datareader import data as pdr
from datetime import date, datetime
import sys
from tqdm import tqdm
from alive_progress import alive_bar
import time
import pathlib

# TODO: Function that returns the dataset in pandas with index Date and without index Date
# TODO: Implement better the folder creation

class StockData():

    def __init__(self, 
                download=False,
                download_stock='AAPL',
                start=None,
                end=None,
                update=False,
                update_stock=None,
                delete=False,
                delete_stock=None,
                range=0,
                folder="../../stock/"):

        self._download = download,
        self._download_stock = download_stock
        self._start = start
        self._end = end
        self._update = update
        self._update_stock = update_stock
        self._delete = delete
        self._delete_stock = delete_stock
        self._range = range
        self._folder = folder

        if self._range > 0:
            self.stock_market = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
            self.stock_market = self.stock_market[:self._range]
        else:
            self.stock_market = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

        self.__main()

    def __download(self):
        yf.pdr_override()

        if self._end == "today" or self._end is None:
            self._end = date.today().strftime("%Y-%m-%d")
        if self._start is None:
            self._start = "2017-01-01"

        if self._download_stock is not None:
            self._download_stock = self._download_stock.upper()
            if self._download_stock.find('.') != -1:
                self._download_stock = self._download_stock.replace('.', '-')     
                
            stock_data = pdr.get_data_yahoo(self._download_stock, 
                                            start=self._start, 
                                            end=self._end,
                                            progress=False,
                                            threads=True)
            stock_data.to_csv(self._folder + self._download_stock + ".csv", \
                                index=True)

        else: 
            with alive_bar(total=self.stock_market.size, title='Download') as bar:
                for i in self.stock_market:
                    bar()
                    if i.find('.') != -1:
                        i = i.replace('.', '-')
                    stock_data = pdr.get_data_yahoo(i, 
                                                start=self._start, 
                                                end=self._end,
                                                progress=False,
                                                threads=True)
                    stock_data.to_csv(self._folder + i + ".csv", 
                                                index=True)

    def download_stock(self):
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
        with alive_bar(total=len(files),title="Deleting") as bar:
            for f in files:
                os.remove(f)

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
                f = f.split('.', 2)[2]
                f = f.split('.', 1)[0]
            f = f.replace('-', '.')
            stock_market.append(os.path.basename(f))

        with alive_bar(total=len(files), title='Updating') as bar:
            for file in stock_market:
                bar() 
                self._update_stock = file
                self.__update_stock()     

    def __update_stock(self):

        if self._end == "today" or self._end is None:
            self._end = date.today().strftime("%Y-%m-%d")

        self._update_stock = self._update_stock.upper()

        if self._update_stock.find('.') != -1:
            self._update_stock = self._update_stock.replace('.', '-')

        self._update_stock = self._update_stock.split('.', 1)[0]

        df_1 = pd.read_csv(self._folder + self._update_stock + '.csv', 
                            parse_dates=False, index_col='Date')
                            
        if df_1.size != 0:
            if self._start is None:
                start = datetime.strptime(pd.to_datetime(df_1.index[0]).strftime("%Y-%m-%d"),
                                "%Y-%m-%d")
            else:
                start = datetime.strptime(self._start, "%Y-%m-%d")
            end = datetime.strptime(self._end, "%Y-%m-%d")
            
            yf.pdr_override()

            df = pdr.get_data_yahoo(self._update_stock,
                                    start=start,
                                    end=end,
                                    progress=False,
                                    threads=True)

            df.to_csv(self.cli_args.folder + self._update_stock + '.csv', 
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
                exit()
        else:
            if self._download is True:
                path = self._folder 
                os.mkdir(self._folder)
                self.__download()
                exit()
        
        if self._download_stock is not None:
            self.download_stock()

        if self._update is True:
            self.update()
            exit()

        if self._update_stock is not None:
            self.__update_stock()

        if self._delete_stock is not None:
            self.__delete_stock()

        dir = os.listdir(path)
        if len(dir) == 0:
            var = input("The directory is empty, you want to download all stock market? [Y/n]")
            if var.lower() == 'y':
                self.create_dataset()
            else:
                raise Exception("If you want to download one stock use --download-stock action")



    