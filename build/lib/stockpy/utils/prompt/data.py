import argparse
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from util.StockDataset import StockData

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
                        default="../../stock/",
                        )
   
    cli_args = parser.parse_args()

    if cli_args.stock_list is not None:
        stock_list = [item for item in cli_args.stock_list.split(',')]

    if cli_args.stock_list is not None:
        StockData(download=cli_args.download,
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
        StockData(download=cli_args.download,
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
