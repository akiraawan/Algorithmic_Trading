import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class Backtest:
    def __init__(self, symbol_file, yahoo=True):
        if yahoo:
            self.symbol = symbol_file
            self.df = yf.download(self.symbol, start="2006-01-01", end="2022-09-06")[["Close"]]
        else:
            file = symbol_file
            self.df = pd.read_csv(file, names=["Date", "Price"], header=None, index_col=0, dayfirst=True)
        
        if self.df.empty:
            print("No data pulled")
        else:
            self.calc_indicators()
            self.generate_signals()
            self.profit = self.calc_metric()

    def calc_indicators(self, SMA=30, devs=(2, 2)):
        self.df["SMA"] = self.df.Close.rolling(SMA).mean()
        self.df["Lower_bb"] = self.df.SMA - self.df.Close.rolling(SMA).std() * devs[0]
        self.df["Upper_bb"] = self.df.SMA + self.df.Close.rolling(SMA).std() * devs[1]
        self.df["distance"] = self.df.Close - self.df.SMA
        self.df.dropna(inplace = True)
    
    def generate_signals(self):
        self.df["position"] = np.where(self.df.Close < self.df.Lower_bb, 1, np.nan)
        self.df["position"] = np.where(self.df.Close > self.df.Upper_bb, -1, self.df["position"])
        self.df["position"] = np.where(self.df.distance * self.df.distance.shift(1) < 0, 0, self.df.position)
        self.df["position"] = self.df.position.ffill().fillna(0)
        self.df.loc[self.df.index[-1], "position"] = 0
        self.df["Trade_Signal"] = self.df.position- self.df.position.shift(1)

    def calc_metric(self, metric="PnL"):
        if metric=="PnL":
            self.df["PnL"] = np.nan
            current_position = 0
            position_price = 0

            for i in self.df.index:
                if (self.df.loc[i, "position"] == 1.0) & (current_position == 0):
                    current_position = 1
                    position_price = self.df.loc[i, "Close"]
                elif (self.df.loc[i, "position"] == -1.0) & (current_position == 0):
                    current_position = -1
                    position_price = self.df.loc[i, "Close"]
                elif self.df.loc[i, "position"] == 0.0:
                    if current_position == 1:
                        PnL = self.df.loc[i, "Close"] - position_price
                        self.df.loc[i, "PnL"] = PnL
                        current_position = 0
                    elif current_position == -1:
                        PnL = position_price - self.df.loc[i, "Close"]
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

    def plot(self, start, cols=None, end=None, format='-'):
        if cols==None:
            cols = ["Close"]
        self.df[cols].loc[start:end].plot(figsize=(12,8))
        plt.show()
