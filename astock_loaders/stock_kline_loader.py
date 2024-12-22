from .stock_base_loader import StockBaseLoader
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from PIL import Image


class StockKlineLoader(StockBaseLoader):
    def __init__(self, symbol, show_chart = False, figsize=(3, 3), categtory_nums = None, **kwargs):
        self.show_chart = show_chart
        self.figsize = figsize
        if categtory_nums is None:
            categtory_nums = (5, 2, -2, -5)
        self.intervals = categtory_nums
        super().__init__(symbol, **kwargs)
        
    def set_feature_cols(self):
        self.feature_cols = self.price_cols + self.other_cols + ['日期']
        return self
        
    def cal_features_labels(self, i, row_data, data):
        feature_start = i
        feature_end = i + self.sequence_length - 1
        label_start = feature_end
        label_end = label_start + self.predict_length - 1

        x_df = data.loc[feature_start:feature_end, self.feature_cols].copy()
        y_df = data.loc[label_start:label_end, self.label_cols].copy()

        max_price = x_df['最高'].max()
        min_price = x_df['最低'].min()

        # Normalize prices
        x_df.loc[:, self.price_cols] = x_df[self.price_cols].apply(
            lambda x: self.normalize_price(max_price, min_price, x)
        )
        # Normalize other features
        x_df.loc[:, self.other_cols] = x_df[self.other_cols].apply(self.min_max_normalize)
        y_df.loc[:, self.label_cols] = y_df[self.label_cols].apply(
            lambda x: self.normalize_price(max_price, min_price, x)
        )

        # 计算涨幅百分比
        pre_price = x_df.iloc[-1]['收盘']
        # 检查 pre_price 是否为零
        if pre_price == 0:
            price_change_percentage = y_df['收盘'].mean() * 100
        else:
            price_change_percentage = ((y_df['收盘'].mean() - pre_price) / pre_price) * 100

        intervals = self.intervals
        result = np.zeros(5)  # 创建一个零数组来存类别概率
        # 使用任何方法对于Series的比较
        if price_change_percentage >= intervals[0]:
            result[0] = 1
        elif price_change_percentage >= intervals[1]:
            result[1] = 1
        elif price_change_percentage >= intervals[2]:
            result[2] = 1
        elif price_change_percentage >= intervals[3]:
            result[3] = 1
        else:
            result[4] = 1

        img_array = self.generate_kline_image(x_df)
        return img_array, result


    
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
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
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
        if self.show_chart:
            plt.show()
        plt.close(fig)
        return img_array / 255.0

    
