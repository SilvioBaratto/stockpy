import argparse
import glob
import io
import os
import pathlib
import urllib.request
from datetime import date

import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_DEFAULT_START = "2017-01-01"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _normalize_symbol(sym: str) -> str:
    """Normalize a ticker symbol to Yahoo Finance form.

    Yahoo Finance uses ``-`` in tickers (e.g. ``BRK-B``) while the Wikipedia
    S&P 500 table uses ``.`` (``BRK.B``).
    """
    return sym.upper().replace(".", "-").replace("=", "").replace(" ", "")


def _fetch_sp500_symbols():
    """Scrape S&P 500 ticker list from Wikipedia.

    Wikipedia rejects the default urllib User-Agent with HTTP 403, so the
    request is made with an explicit browser UA and the HTML is fed into
    ``pd.read_html`` from memory.
    """
    req = urllib.request.Request(_SP500_URL, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    return pd.read_html(io.StringIO(html))[0]["Symbol"]


class DataDownloader:
    def __init__(
        self,
        download=False,
        download_stock=None,
        start=None,
        end=None,
        update=False,
        update_stock=None,
        delete=False,
        delete_stock=None,
        range=0,
        folder="../../stock/",
    ):
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

        if isinstance(self._download_stock, list):
            self.stock_market = [_normalize_symbol(s) for s in self._download_stock]
        elif self._range > 0:
            syms = _fetch_sp500_symbols()
            self.stock_market = [_normalize_symbol(s) for s in syms[: self._range]]
        elif self._download is True or (
            self._download_stock is None
            and not self._update
            and self._update_stock is None
            and not self._delete
            and self._delete_stock is None
        ):
            syms = _fetch_sp500_symbols()
            self.stock_market = [_normalize_symbol(s) for s in syms]
        else:
            self.stock_market = []

        self.__main()

    def __download(self):
        if self._end in (None, "today"):
            self._end = date.today().strftime("%Y-%m-%d")
        if self._start is None:
            self._start = _DEFAULT_START

        if isinstance(self._download_stock, str):
            sym = _normalize_symbol(self._download_stock)
            df = yf.download(
                sym,
                start=self._start,
                end=self._end,
                auto_adjust=True,
                progress=True,
                threads=False,
                ignore_tz=False,
                multi_level_index=False,
            )
            if df is not None and not df.empty:
                df.to_csv(os.path.join(self._folder, f"{sym}.csv"), index=True)
            return

        df = yf.download(
            self.stock_market,
            start=self._start,
            end=self._end,
            auto_adjust=True,
            progress=True,
            threads=True,
            ignore_tz=False,
            group_by="ticker",
        )
        if df is None or df.empty:
            return

        top_level = (
            df.columns.get_level_values(0)
            if isinstance(df.columns, pd.MultiIndex)
            else df.columns
        )
        for sym in tqdm(self.stock_market, leave=False):
            if sym not in top_level:
                continue
            sub = df[sym].dropna(how="all")
            if sub.empty:
                continue
            sub.to_csv(os.path.join(self._folder, f"{sym}.csv"), index=True)

    def download_stock(self):
        if isinstance(self._download_stock, list):
            for sym in self._download_stock:
                path = os.path.join(self._folder, f"{_normalize_symbol(sym)}.csv")
                if not os.path.isfile(path) or len(os.listdir(self._folder)) == 0:
                    return self.__download()
                check = input(
                    "This file already exist, do you want download proceed [y/n]: "
                )
                if check.lower() == "y":
                    return self.__download()
                raise Exception("download failed")
        else:
            assert isinstance(self._download_stock, str)
            sym = _normalize_symbol(self._download_stock)
            path = os.path.join(self._folder, f"{sym}.csv")
            if not os.path.isfile(path) or len(os.listdir(self._folder)) == 0:
                return self.__download()
            check = input(
                "This file already exist, do you want download proceed [y/n]: "
            )
            if check.lower() == "y":
                return self.__download()
            raise Exception("download failed")

    def __delete(self):
        files = glob.glob(os.path.join(self._folder, "*"))
        with tqdm(total=len(files), leave=False) as bar:
            for f in files:
                os.remove(f)
                bar.update()

    def __delete_stock(self):
        assert self._delete_stock is not None
        sym = _normalize_symbol(self._delete_stock)
        path = os.path.join(self._folder, f"{sym}.csv")
        if os.path.isfile(path):
            os.remove(path)
        else:
            print("Error: %s file not found" % path)

    def update(self):
        files = glob.glob(os.path.join(self._folder, "*.csv"))
        with tqdm(total=len(files), leave=False) as bar:
            for f in files:
                bar.update()
                self._update_stock = os.path.splitext(os.path.basename(f))[0]
                self.__update_stock()

    def __update_stock(self):
        assert self._update_stock is not None
        sym = _normalize_symbol(self._update_stock)
        path = os.path.join(self._folder, f"{sym}.csv")
        if not os.path.isfile(path):
            print("Error: %s file not found" % path)
            return

        df_old = pd.read_csv(path, parse_dates=True, index_col="Date")

        end = (
            date.today().strftime("%Y-%m-%d")
            if self._end in (None, "today")
            else self._end
        )

        if not df_old.empty and self._start is None:
            start = (df_old.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = self._start or _DEFAULT_START

        if start >= end:
            if not self._update:
                print(f"{sym}.csv already up-to-date")
            return

        df_new = yf.download(
            sym,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            ignore_tz=True,
            multi_level_index=False,
        )

        if df_new is None or df_new.empty:
            return

        merged = pd.concat([df_old, df_new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_csv(path, index_label="Date")

        if not self._update:
            print(f"{sym}.csv Downloaded")

    def __main(self):
        path = pathlib.Path(self._folder)

        if self._delete:
            self.__delete()
            print("deleted all files from ", path)
            exit()

        if not path.exists():
            os.makedirs(self._folder, exist_ok=True)

        if self._download is True:
            self.__download()

        if self._download_stock is not None:
            self.download_stock()

        if self._update is True:
            self.update()

        if self._update_stock is not None:
            self.__update_stock()

        if self._delete_stock is not None:
            self.__delete_stock()

        if not os.listdir(self._folder):
            var = input(
                "The directory is empty, you want to download all stock market? [Y/n]"
            )
            if var.lower() == "y":
                self.__download()
            else:
                raise Exception("If you want to download one stock use --stock action")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download",
        help="download all stocks from S&P",
        action="store_true",
    )
    parser.add_argument(
        "--stock",
        help="Download data from stock market, default is S&P",
        action="store",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--stock-list",
        help="list of stock to download",
        action="store",
        type=str,
    )
    parser.add_argument(
        "--start",
        help="In which day start the download, default 2017-01-01",
        action="store",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--end",
        help="Last day of the dataset, default today",
        action="store",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--update",
        help="Update the choosen dataset",
        action="store_true",
    )
    parser.add_argument(
        "--update-stock",
        help="Update a precise stock",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--delete",
        help="Delete all the folder",
        action="store_true",
    )
    parser.add_argument(
        "--delete-stock",
        help="Delete a precise stock",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--range",
        help="First n stocks in range",
        action="store",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--folder",
        help="Destination folder for downloaded CSVs",
        action="store",
        default="stock/",
    )

    cli_args = parser.parse_args()

    download_stock = cli_args.stock
    if cli_args.stock_list is not None:
        download_stock = [item for item in cli_args.stock_list.split(",")]

    DataDownloader(
        download=cli_args.download,
        download_stock=download_stock,
        start=cli_args.start,
        end=cli_args.end,
        update=cli_args.update,
        update_stock=cli_args.update_stock,
        delete=cli_args.delete,
        delete_stock=cli_args.delete_stock,
        range=cli_args.range,
        folder=cli_args.folder,
    )


if __name__ == "__main__":
    main()
