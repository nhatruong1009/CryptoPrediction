from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX, AutoReg, ExponentialSmoothing, SimpleExpSmoothing, Holt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense 

class ArimaEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, order=(5,1,0)):
        self.order = order
    def fit(self, X, y):
        return self
    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            self.model = ARIMA(X[i], order=self.order)
            self.model_fit = self.model.fit()
            pred.append(self.model_fit.forecast()[0])
        return pred
    
class SarimaxEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, order=(5,1,0), seasonal_order=(1,1,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order
    def fit(self, X, y):
        return self
    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            self.model = SARIMAX(X[i], order=self.order, seasonal_order=self.seasonal_order)
            self.model_fit = self.model.fit()
            pred.append(self.model_fit.forecast()[0])
        return pred
    
class AutoRegEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, lags=1):
        self.lags = lags
    def fit(self, X, y):
        return self
    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            self.model = AutoReg(X[i], lags=self.lags)
            self.model_fit = self.model.fit()
            pred.append(self.model_fit.predict(len(X[i]), len(X[i]))[0])
        return pred
    
class ExponentialSmoothingEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, trend=None, damped_trend=False, seasonal=None, seasonal_periods=12):
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
    def fit(self, X, y):
        return self
    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            self.model = ExponentialSmoothing(X[i], trend=self.trend, damped_trend=self.damped_trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
            self.model_fit = self.model.fit()
            pred.append(self.model_fit.forecast()[0])
        return pred

class HoltEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        return self
    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            self.model = Holt(X[i])
            self.model_fit = self.model.fit()
            pred.append(self.model_fit.forecast()[0])
        return pred
    
class LSTMEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=2, batch_size=1, neurons=200):
        self.epochs = epochs
        self.batch_size = batch_size
        self.neurons = neurons
    def fit(self, X, y):
        self.model = Sequential()
        self.model.add(LSTM(self.neurons, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(LSTM(self.neurons, return_sequences=False))
        self.model.add(Dense(100))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)
        return self
    def predict(self, X):
        return self.model.predict(X)