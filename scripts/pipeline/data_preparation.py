from data_imputer import data_imputer
from category_encoder import category_encoder
from data_normalization import data_nomalization
from outlier_remover import outlier_remover
from feature_selector import feature_selector
from feature_engineering import feature_engineering

# Classe principale che coordina tutte le trasformazioni
class data_preparation:
    def __init__(self):
        self.imputer = data_imputer()
        self.encoder = category_encoder()
        self.normalizer = data_nomalization()
        self.outlier_remover = outlier_remover()
        self.feature_selector = feature_selector()
        self.engineer = feature_engineering()


    def fit(self, X_train, y_train):
        # Fase di fit per tutte le trasformazioni
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_imputed = self.engineer.transform(X_train_imputed)
        y_train = y_train.loc[X_train_imputed.index]
        X_train_encoded = self.encoder.fit_transform(X_train_imputed, y_train)
        X_train_normalized = self.normalizer.fit_transform(X_train_encoded)
        X_train_no_outliers = self.outlier_remover.transform(X_train_normalized)
        self.feature_selector.fit(X_train_no_outliers)
        return X_train_no_outliers

    def transform_train(self, X_train):
        X_train_final = self.feature_selector.transform(X_train)
        return X_train_final

    def transform_test(self, X_test):
        # Applica le trasformazioni al test set (escluso outlier removal)
        X_test_imputed = self.imputer.transform(X_test)
        X_test_imputed = self.engineer.transform(X_test_imputed)
        X_test_encoded = self.encoder.transform(X_test_imputed)
        X_test_normalized = self.normalizer.transform(X_test_encoded)
        X_test_final = self.feature_selector.transform(X_test_normalized)
        return X_test_final