from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

# Classe per la codifica delle categorie
class category_encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.catboost_encoder = ce.CatBoostEncoder()
        self.marca_means = None
        self.global_mean = None
        self.known_makes = None
        self.known_models = None
        self.cat_cols = None

    def fit(self, X, y) -> 'category_encoder':

        if 'trasmissione' in X.columns:
            # Codifica 'trasmissione' con LabelEncoder
            self.label_encoder.fit(X['trasmissione'])

        # Colonne categoriche da codificare con CatBoostEncoder
        self.cat_cols = [col for col in X.select_dtypes(include=['object', 'category']).columns if col != 'trasmissione']

        # Fit CatBoostEncoder
        self.catboost_encoder.fit(X[self.cat_cols], y)

        # Prepara sostituzioni per modelli non visti
        self.marca_means = X.join(y.rename('target')).groupby('marca')['target'].mean()
        self.global_mean = y.mean()
        self.known_makes = X['marca'].unique()
        self.known_models = X['modello'].unique()

        return self

    def transform(self, X):
        X = X.copy()

        if 'trasmissione' in X.columns:
            # Applica LabelEncoder a 'trasmissione'
            X['trasmissione'] = self.label_encoder.transform(X['trasmissione'])

        # Applica CatBoostEncoder alle altre colonne categoriche
        encoded = self.catboost_encoder.transform(X[self.cat_cols])

        # Gestisci modelli non presenti nel training set
        new_models_mask = ~X['modello'].isin(self.known_models)
        if new_models_mask.any():
            replacements = X.loc[new_models_mask, 'marca'].map(self.marca_means).fillna(self.global_mean)
            encoded.loc[new_models_mask, 'modello'] = replacements.values

        X[self.cat_cols] = encoded
        return X
