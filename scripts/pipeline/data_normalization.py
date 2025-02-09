from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Classe per la normalizzazione
class data_nomalization(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = None

    def fit(self, X):
        self.numeric_cols = X.select_dtypes(include=['number']).columns
        self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        X = X.copy()
        numeric_cols = self.numeric_cols.tolist()

        X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        return X