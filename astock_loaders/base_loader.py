from sklearn.model_selection import train_test_split
import numpy as np
from .stock_dataset import StockDataset
from torch.utils.data import DataLoader

class BaseLoader:
    def __init__(self, sequence_length=30, predict_length=1, batch_size=32, test_ratio=0.1):
        self.predict_length = predict_length
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        
        
    def cal_features_labels(self, i, row_data, data):
        """
        计算特征值

        :param i: 当前行索引
        :param row_data: 当前行数据
        :param data: 整个数据集
        :return: 特征值
        """
        raise NotImplementedError("Subclasses must implement the cal_features method.")
    
    def create_sequences(self):
        if not hasattr(self, 'data'):
            self.format_data()
        X, Y = [], []
        length = len(self.data)
        for i in range(length - self.sequence_length - self.predict_length):
            x, y = self.cal_features_labels(i, self.data.iloc[i], self.data)
            X.append(x)
            Y.append(y)

        arr_x = np.array(X, dtype=np.float32)
        arr_y = np.array(Y, dtype=np.float32)
        return (arr_x, arr_y)

    def get_dataset(self):
        if not hasattr(self, 'train_dataset') or not hasattr(self, 'test_dataset'):
            X, Y = self.create_sequences()
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_ratio, shuffle=False)
            self.train_dataset = StockDataset(X_train, Y_train)
            self.test_dataset = StockDataset(X_test, Y_test)
        return self.train_dataset, self.test_dataset

    def get_train_loader(self):
        train_dataset,_ = self.get_dataset()
        return DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 4)

    def get_test_loader(self):
        _ ,test_dataset = self.get_dataset()
        return DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 4)
    
    def get_recent_data(self):
        if not hasattr(self, 'data'):
            self.format_data()
        idx = len(self.data) - self.sequence_length
        x_df, _ = self.cal_features_labels(idx, self.data.iloc[idx], self.data)
        return np.array(x_df, dtype=np.float32)
        
    
