from .stock_trend_loader import StockTrendLoader
from .base_loader import BaseLoader
from china_stock_data import StockMarket
from .utils import is_a_share
class IndexTrendLoader:
    def __init__(self, symbols = None, index = None, start = None, limit = None, **kwargs):
        self.stocks: dict[str, BaseLoader] = {}
        self.kwargs = kwargs
        if symbols is not None:
            self.symbols = symbols
            for symbol in symbols:
                self.stocks[symbol] = StockTrendLoader(symbol, **kwargs)
        if index:
            self.stock_market = StockMarket(index)
            codes = self.stock_market['index_codes']

            if start is None:
                start = 0
            if limit is None:
                limit = len(codes)

            actual_limit = min(start + limit, len(codes))

            for i in range(start, actual_limit):
                symbol = codes[i]
                if symbol not in self.stocks and is_a_share(symbol):
                    self.stocks[symbol] = StockTrendLoader(symbol=symbol, **kwargs)
                    
