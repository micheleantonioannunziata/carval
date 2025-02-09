# Classe per la rimozione degli outlier
class outlier_remover:
    def __init__(self, threshold=3):
        self.threshold = threshold

    def transform(self, X):
        numeric_cols = X.select_dtypes(include=['number']).columns
        z_scores = (X[numeric_cols] - X[numeric_cols].mean()) / X[numeric_cols].std()
        mask = (z_scores.abs() <= self.threshold).all(axis=1)
        return X[mask]