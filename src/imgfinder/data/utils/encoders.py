import numpy as np

class LabelEncoder:
    def __init__(self) -> None:
        self.transform_conv: dict = None
        self.inverse_transform_conv: dict = None
        
        self.transform = np.vectorize(self.__transform)
        self.inverse_transform = np.vectorize(self.__inverse_transform)
        
    def fit(self, y):
        self.transform_conv = dict()
        self.inverse_transform_conv = dict()
        for key, value in enumerate(np.unique(y)):
            self.inverse_transform_conv[key] = value
            self.transform_conv[value] = key
    
    def __transform(self, y):
        return self.transform_conv[y]
    
    def __inverse_transform(self, y):
        return self.inverse_transform_conv[y]
    
    def fit_transform(self, y):
        self.fit(y)
        
        return self.transform(y)
        
        