import numpy as np
import pandas as pd
import os 
import glob
import argparse
import yfinance as yf
from pandas_datareader import data as pdr
import pathlib
from tqdm.auto import tqdm
from datetime import date, datetime
from typing import Union, Tuple

class DataDownloader():

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

def main():
    parser = argparse.ArgumentParser()

        # Feature implementation:
        # Select to download stocks from different stock-market than S&P

    parser.add_argument('--download',
                        help="download all stocks from S&P",
                        action="store_true",
                        )

    parser.add_argument('--stock',
                        help='Download data from stock market, default is S&P',
                        action='store',
                        default=None,
                        type=str
                        )

    parser.add_argument('--stock-list',
                        help="list of stock to download",
                        action="store",
                        type=str,
                        )

    parser.add_argument('--start',
                        help='In which day start the download, default 2017-01-01',
                        action='store',
                        default=None,
                        type=str
                        )
    parser.add_argument('--end',
                        help='Last day of the dataset, default today',
                        action='store',
                        default=None,
                        type=str
                        )
    parser.add_argument('--update',
                        help="Update the choosen dataset",
                        action='store_true',
                        )
    parser.add_argument('--update-stock',
                        help="Delete a precise stock",
                        action='store',
                        default=None,
                        )
    parser.add_argument('--delete',
                        help="Delete all the folder",
                        action="store_true"
                        )
    parser.add_argument('--delete-stock',
                        help="Delete a precise stock",
                        action='store',
                        default=None
                        )
    parser.add_argument('--range',
                        help="First n stocks in range",
                        action='store',
                        default=0,
                        type=int,
                        )
    parser.add_argument('--folder',
                        help="First n stocks in range",
                        action='store',
                        default="stock/",
                        )
   
    cli_args = parser.parse_args()

    if cli_args.stock_list is not None:
        stock_list = [item for item in cli_args.stock_list.split(',')]

    if cli_args.stock_list is not None:
        DataDownloader(download=cli_args.download,
                download_stock=stock_list,
                start=cli_args.start,
                end=cli_args.end,
                update=cli_args.update,
                update_stock=cli_args.update_stock,
                delete=cli_args.delete,
                delete_stock=cli_args.delete_stock,
                range=cli_args.range,
                folder=cli_args.folder)
    else:
        DataDownloader(download=cli_args.download,
                download_stock=cli_args.stock,
                start=cli_args.start,
                end=cli_args.end,
                update=cli_args.update,
                update_stock=cli_args.update_stock,
                delete=cli_args.delete,
                delete_stock=cli_args.delete_stock,
                range=cli_args.range,
                folder=cli_args.folder)     

if __name__ == "__main__":
    main()
