import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
plt.style.use("seaborn")
'''
Object-Oriented Backtesting System for Mean Reversion Strategy

'''
class Backtest:
    def __init__(self, symbol_file, yahoo=True):
        if yahoo:
            self.symbol = symbol_file
            self.df = yf.download(self.symbol, start="2006-01-01", end="2022-09-06")[["Close"]]
            self.df.columns = ["Price"]
        else:
            file = symbol_file
            self.df = pd.read_csv(file, names=["Date", "Price"], header=None, index_col=0, dayfirst=True)
            self.df.index = pd.to_datetime(self.df.index, dayfirst=True)
        
        if self.df.empty:
            print("No data pulled")
        else:
            self.calc_indicators()
            self.generate_signals()
            self.profit = self.calc_metric()

    def calc_indicators(self, SMA=30, devs=(2, 2)):
        self.df["SMA"] = self.df.Price.rolling(SMA).mean()
        self.df["Lower_bb"] = self.df.SMA - self.df.Price.rolling(SMA).std() * devs[0]
        self.df["Upper_bb"] = self.df.SMA + self.df.Price.rolling(SMA).std() * devs[1]
        self.df["distance"] = self.df.Price - self.df.SMA
        self.df.dropna(inplace = True)
    
    def generate_signals(self):
        self.df["position"] = np.where(self.df.Price < self.df.Lower_bb, 1, np.nan)
        self.df["position"] = np.where(self.df.Price > self.df.Upper_bb, -1, self.df["position"])
        self.df["position"] = np.where(self.df.distance * self.df.distance.shift(1) < 0, 0, self.df.position)
        self.df["position"] = self.df.position.ffill().fillna(0)
        self.df.loc[self.df.index[-1], "position"] = 0
        self.df["Trade_Signal"] = self.df.position - self.df.position.shift(1)

    def calc_metric(self, metric="PnL"):
        if metric=="PnL":
            self.df["PnL"] = np.nan
            current_position = 0
            position_price = 0

            for i in self.df.index:
                if (self.df.loc[i, "position"] == 1.0) & (current_position == 0):
                    current_position = 1
                    position_price = self.df.loc[i, "Price"]
                elif (self.df.loc[i, "position"] == -1.0) & (current_position == 0):
                    current_position = -1
                    position_price = self.df.loc[i, "Price"]
                elif self.df.loc[i, "position"] == 0.0:
                    if current_position == 1:
                        PnL = self.df.loc[i, "Price"] - position_price
                        self.df.loc[i, "PnL"] = PnL
                        current_position = 0
                    elif current_position == -1:
                        PnL = position_price - self.df.loc[i, "Price"]
                        self.df.loc[i, "PnL"] = PnL
                        current_position = 0

            self.df["cumPnL"] = self.df["PnL"].cumsum()
            metric = self.df["PnL"].sum()
        
        elif metric=="log_return":
            self.df["log_returns"] = np.log(self.df.div(self.df.shift(1)))
            self.df["log_strategy"] = self.df.position.shift(1) * self.df["log_returns"]
            self.df.dropna(inplace = True)

            metric = np.exp(self.df["log_strategy"].sum())

        return metric

    def plot(self, start, end=None, cols=None, format='-', with_signal=False):
        if cols==None:
            cols = ["Price"]
        ax = self.df[cols].loc[start:end].plot(figsize=(12,8))
        if with_signal:
            buysignals = self.df[self.df["Trade_Signal"]==1.0].loc[start:end].index
            sellsignals = self.df[self.df["Trade_Signal"]==-1.0].loc[start:end].index
            
            plt.plot(buysignals, self.df.loc[buysignals, "Price"], "g^")
            plt.plot(sellsignals, self.df.loc[sellsignals, "Price"], "rv")
        plt.show()

    def plot_PnL(self, start, end=None, cumulative=False, format='-'):
        if not cumulative:
            self.df.dropna()["PnL"].loc[start:end].plot(figsize=(12,8))
        else:
            self.df.dropna()["cumPnL"].loc[start:end].plot(figsize=(12,8))