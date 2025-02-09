class feature_selector:
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.features_to_keep = None

    def fit(self, X):
        variances = X.var()
        self.features_to_keep = variances[variances > self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X[self.features_to_keep]