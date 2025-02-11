import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from data_preparation import data_preparation
from pre_process import pre_process


class CarValPipeline:
    def __init__(self, n_splits=5):
        self.output_dir = '../../diagrams'
        self.data_preparator = data_preparation()
        self.pre_processor = pre_process()
        self.n_splits = n_splits
        self.results = []
        self.model = None

        # Creazione cartella per salvare i diagrammi
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_data(self, data_path):
        print('Lettura dataset...')
        df = pd.read_csv(data_path, on_bad_lines='skip')

        print('Pre-processing dataset...')
        new_path = self.pre_processor.preprocess(df)

        print('Pre-processing completato')
        df = pd.read_csv(new_path, on_bad_lines='skip')

        # Separazione in features e target
        X = df.drop(columns=['prezzo'])
        y = df['prezzo']

        return X, y

    def train(self, X_train, X_test, y_train, y_test, fold):
        print(f'\nTraining del modello per fold {fold + 1}/{self.n_splits}...')
        model = RandomForestRegressor(
        max_depth=18,                  # Controlla la complessità del modello
        n_estimators=300,               # Bilanciamento tra performance e velocità
        min_samples_split=10,            # Riduce overfitting
        min_samples_leaf=4,             # Previene overfitting sulle foglie
        max_features='sqrt',            # Ottimizza il trade-off bias-variance
        n_jobs=-1,                      # Usa tutti i core disponibili
        random_state=20,
        bootstrap=True,                 # Migliora la generalizzazione
        criterion='squared_error'
        )

        model.fit(X_train, y_train)

        # Previsioni
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calcolo delle metriche
        fold_results = {
            'Fold': fold + 1,
            'MAE_train': mean_absolute_error(y_train, y_train_pred),
            'MAE_test': mean_absolute_error(y_test, y_test_pred),
            'MSE_train': mean_squared_error(y_train, y_train_pred),
            'MSE_test': mean_squared_error(y_test, y_test_pred),
            'RMSE_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'RMSE_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAPE_train': mean_absolute_percentage_error(y_train, y_train_pred),
            'MAPE_test': mean_absolute_percentage_error(y_test, y_test_pred),
        }

        self.results.append(fold_results)

        self.model = model

        # Plot Feature Importance
        feature_importances = model.feature_importances_
        feature_names = X_train.columns

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances, color='skyblue')
        plt.xlabel('Importanza')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - Fold {fold + 1}')
        os.makedirs(os.path.join(self.output_dir, 'feature_importances'), exist_ok=True)
        plt.savefig(f'{self.output_dir}/feature_importances/feature_importance_fold_{fold + 1}.png')
        plt.close()

    def save_results(self):
        # Salviamo i risultati in un file CSV
        results_df = pd.DataFrame(self.results)
        os.makedirs('./results', exist_ok=True)
        results_df.to_csv('./results/fold_results.csv', index=False)

    def run_pipeline(self, path):
        X, y = self.prepare_data(path)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Data preparation per ogni fold
            print(f'\nData preparation per il fold {fold + 1}/{self.n_splits}...')
            X_train = self.data_preparator.fit(X_train, y_train)
            X_train = self.data_preparator.transform_train(X_train)
            X_test = self.data_preparator.transform_test(X_test)

            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]

            self.train(X_train, X_test, y_train, y_test, fold)

        # Salva i risultati ad ogni fold
        self.save_results()

        # Media delle metriche finali
        final_results = pd.DataFrame(self.results).mean()

        print('\nRisultati finali mediati su tutti i fold:')
        print(final_results)

        # Salvataggio del modello finale
        joblib.dump(self.model, 'random_forest_regressor_model.pkl')
        joblib.dump({'preparator': self.data_preparator}, 'pipeline_regressor_transformers.pkl')


if __name__ == "__main__":
    pipe = CarValPipeline(n_splits=10)  # Usa 10-fold cross-validation
    path = "../../dataset/car_prices.csv"
    pipe.run_pipeline(path)
