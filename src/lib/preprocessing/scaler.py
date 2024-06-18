from numpy import ndarray
from pandas.core.series import Series
from sklearn.preprocessing import StandardScaler as SKStandardScaler, MinMaxScaler as SKMinMaxScaler



class StandardScaler(SKStandardScaler):
    def __init__(self, *, copy: bool = True, with_mean: bool = True, with_std: bool = True, total_mode: bool = False) -> None:
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        self.total_mode = total_mode
    
    def fit(self, X , y: Series | ndarray | None = None) -> SKMinMaxScaler:
        if self.total_mode:
            X = X.flatten().reshape(-1, 1)
        return super().fit(X, y)
    
    def transform(self, X) -> ndarray:
        if self.total_mode:
            X = X.flatten().reshape(-1, 1)
        res = super().transform(X)
        if self.total_mode:
            res = res.flatten()
        return res



class MinMaxScaler(SKMinMaxScaler):
    def __init__(self, feature_range: tuple[int, int] | list[int] = (0,1), *, copy: bool = True, clip: bool = False, total_mode: bool = False) -> None:
        if isinstance(feature_range, list):
            feature_range = tuple(feature_range)
        self.total_mode = total_mode
        super().__init__(feature_range, copy=copy, clip=clip)

    def fit(self, X , y: Series | ndarray | None = None) -> SKMinMaxScaler:
        if self.total_mode:
            X = X.flatten().reshape(-1, 1)
        return super().fit(X, y)


    def transform(self, X) -> ndarray:
        if self.total_mode:
            X = X.flatten().reshape(-1, 1)
        res = super().transform(X)
        if self.total_mode:
            res = res.flatten()
        return res
    
    def inverse_transform(self, X) -> ndarray:
        return super().inverse_transform(X)


