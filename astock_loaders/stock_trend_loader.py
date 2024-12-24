from china_stock_data import StockData
from .stock_base_loader import StockBaseLoader
import math
class StockTrendLoader(StockBaseLoader):
    def __init__(self, symbol, **kwargs):
        super().__init__(symbol, **kwargs)
        
    def set_feature_cols(self):
        self.feature_cols = self.price_cols + self.other_cols
        return self

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
        return x_df, y_df
    
    
    
