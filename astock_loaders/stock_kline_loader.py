from china_stock_data import StockData
from .base_loader import BaseLoader
import math
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from PIL import Image


class StockKlineLoader(BaseLoader):
    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.stock = StockData(symbol, days=365 * 10)
        self.price_cols = ['开盘', '收盘', '最高', '最低']
        self.other_cols = ['成交量']
        self.label_cols = ['收盘']
        self.feature_cols = self.price_cols + self.other_cols + ['日期']
        super().__init__(**kwargs)
        
    def origin_data(self):
        """
        从china_stock_data加载原始的数据，这里只加载了历史交易数据，K线数据
        :return: self对象
        """
        self.origin = self.stock.kline
        return self
    
    def format_data(self):
        """
        格式化数据，包括加载原始数据、处理数据、计算最高和最低价格，并将数据存储在self.data中
        :return: self对象
        """
        if not hasattr(self, 'origin'):
            self.origin_data()
        data = self.origin.copy()
        data.drop(['股票代码'], axis=1, inplace=True)
        self.max_price = math.ceil(data['最高'].max())
        self.min_price = math.floor(data['最低'].min())
        data['成交量'] = data['成交量'].astype(float)
        self.data = data
        return self
    
    def normalize_price(self, max_price, min_price, val):
        return (val - min_price) / (max_price - min_price)
    
    def unnormalize_price(self, val):
        return val * (self.max_price - self.min_price) + self.min_price
    
    def min_max_normalize(self, series):
        return (series - series.min()) / (series.max() - series.min())
    
    def cal_features_labels(self, i, row_data, data):
        feature_start = i
        feature_end = i + self.sequence_length - 1
        label_start = feature_end
        label_end = label_start + self.predict_length - 1

        x_df = data.loc[feature_start:feature_end, self.feature_cols].copy()
        y_df = data.loc[label_start:label_end, self.label_cols].copy()
        
        max_price = math.ceil(x_df['最高'].max())
        min_price = math.floor(x_df['最低'].min())

        # Normalize prices
        x_df.loc[:, self.price_cols] = x_df[self.price_cols].apply(
            lambda x: self.normalize_price(max_price, min_price, x)
        )
        # Normalize other features
        x_df.loc[:, self.other_cols] = x_df[self.other_cols].apply(self.min_max_normalize)
        # Normalize labels
        y_df.loc[:, self.label_cols] = y_df[self.label_cols].apply(
            lambda x: self.normalize_price(max_price, min_price, x)
        )

        y_df = y_df.mean()

        # 生成 K 线图并转换为图像数组
        img_array = self.generate_kline_image(x_df)

        return img_array, y_df

    def generate_kline_image(self, df):
        # 确保索引为 DatetimeIndex
        df.index = pd.to_datetime(df['日期'])
        df.rename(columns={
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume'
        }, inplace=True)


        mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='yahoo')
        
        fig, ax = plt.subplots(figsize=(5, 3))
        
        try:
            mpf.plot(df, type='candle', ax=ax, show_nontrading=False, style=s)
        except Exception as e:
            print("Error plotting the chart:", e)  # 捕获并打印错误信息

        buf = BytesIO()
        plt.axis('off')
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.5)
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)
        # plt.show()
        plt.close(fig)
        return img_array


    
    def load_data(self):
        """
        返回类似这样的数据
        :return: (x_train, y_train), (x_test, y_test)
        """
        if not hasattr(self, 'data'):
            self.format_data()
        
            
    def load_recent(self):
        """
        返回类似这样的数据
        :return: (x_train, y_train), (x_test, y_test)
        """
        if not hasattr(self, 'data'):
            self.format_data()     
            
    
    
