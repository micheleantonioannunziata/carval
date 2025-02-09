# Classe per l'imputazione dei dati
from sklearn.base import BaseEstimator, TransformerMixin


class data_imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.group_modes = {}
        self.mean_condizione_by_anno = {}
        self.mean_chilometraggio_by_anno = {}
        self.mean_chilometraggio_by_eta = {}
        self.mean_condizione_by_eta = {}
        self.overall_modes = {}
        self.overall_condizione_mean = None
        self.overall_chilometraggio_mean = None

    def fit(self, X, y = None) -> 'data_imputer':
        # Calcola le mode per gruppo
        self.group_modes['carrozzeria'] = X.groupby('modello')['carrozzeria'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
        self.group_modes['trasmissione'] = X.groupby('modello')['trasmissione'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
        self.group_modes['colorazione'] = X.groupby('marca')['colorazione'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
        self.group_modes['colore interni'] = X.groupby('marca')['colore interni'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
        self.group_modes['modello'] = X.groupby('marca')['modello'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
        self.group_modes['marca'] = X.groupby('modello')['marca'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()

        # Calcola le medie per anno
        self.mean_condizione_by_anno = X.groupby('anno produzione')['condizione'].mean().to_dict()
        self.mean_chilometraggio_by_anno = X.groupby('anno produzione')['chilometraggio'].mean().to_dict()

        self.mean_condizione_by_eta = {
            2015 - anno: media for anno, media in self.mean_condizione_by_anno.items()
        }

        self.mean_chilometraggio_by_eta = {
            2015 - anno: media for anno, media in self.mean_chilometraggio_by_anno.items()
        }

        # Calcola le mode complessive e le medie dopo imputazione
        self.overall_modes = {
            'carrozzeria': X['carrozzeria'].mode()[0],
            'trasmissione': X['trasmissione'].mode()[0],
            'colorazione': X['colorazione'].mode()[0],
            'colore interni': X['colore interni'].mode()[0],
            'modello': X['modello'].mode()[0],
            'marca': X['marca'].mode()[0]
        }

        # Calcola le medie complessive dopo imputazione per anno
        temp_condizione = X['condizione'].fillna(X['anno produzione'].map(self.mean_condizione_by_anno))
        self.overall_condizione_mean = temp_condizione.mean()

        temp_chilometraggio = X['chilometraggio'].fillna(X['anno produzione'].map(self.mean_chilometraggio_by_anno))
        self.overall_chilometraggio_mean = temp_chilometraggio.mean()

        return self

    def transform(self, X):
        X = X.copy()

        # Drop delle righe dove sia "Marca" che "Modello" sono null
        X = X.dropna(subset=["marca", "modello"], how="all")


        if 'trasmissione' in X.columns:
            X['trasmissione'] = X['trasmissione'].fillna(X['modello'].map(self.group_modes['trasmissione'])).fillna(
                self.overall_modes['trasmissione'])

        # Applica l'imputazione basata sui gruppi
        X['marca'] = X['marca'].fillna(X['marca'].map(self.group_modes['marca'])).fillna(self.overall_modes['marca'])
        X['modello'] = X['modello'].fillna(X['modello'].map(self.group_modes['modello'])).fillna(self.overall_modes['modello'])

        X['carrozzeria'] = X['carrozzeria'].fillna(X['modello'].map(self.group_modes['carrozzeria'])).fillna(self.overall_modes['carrozzeria'])

        X['colorazione'] = X['colorazione'].fillna(X['marca'].map(self.group_modes['colorazione'])).fillna(self.overall_modes['colorazione'])
        X['colore interni'] = X['colore interni'].fillna(X['marca'].map(self.group_modes['colore interni'])).fillna(self.overall_modes['colore interni'])

        X['allestimento'] = X['allestimento'].fillna('base')

        if('anno produzione' not in X.columns):
            # Imputa condizione e chilometraggio
            X['condizione'] = X['condizione'].fillna(X['età'].map(self.mean_condizione_by_eta)).fillna(self.overall_condizione_mean)
            X['chilometraggio'] = X['chilometraggio'].fillna(X['età'].map(self.mean_chilometraggio_by_eta)).fillna(self.overall_chilometraggio_mean)
        else:
            X['condizione'] = X['condizione'].fillna(X['anno produzione'].map(self.mean_condizione_by_anno)).fillna(
                self.overall_condizione_mean)
            X['chilometraggio'] = X['chilometraggio'].fillna(X['anno produzione'].map(self.mean_chilometraggio_by_anno)).fillna(
                self.overall_chilometraggio_mean)


        return X