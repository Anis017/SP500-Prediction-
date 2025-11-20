import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from src.data_prep import DataPreprocessor
from src.models import ModelFactory
from src.ensemble import EnsembleModel
from src.probabilistic import ProbabilisticForecaster
from src.evaluation import ModelEvaluator
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Pipeline principal de prédiction du S&P 500"""
    logger.info("Démarrage du projet de prédiction S&P 500")
    
    # 1. Préparation des données
    logger.info("Étape 1: Préparation des données")
    preprocessor = DataPreprocessor()
    
    # Chargement et préparation des données
    try:
        data = preprocessor.load_and_prepare_data()
        logger.info(f"Données chargées: {data.shape}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return
    
    # Feature engineering avancé
    data = preprocessor.create_advanced_features(data)
    logger.info(f"Features créées: {data.shape}")
    
    # Séparation des données
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_features_target(data)
    logger.info(f"Données d'entraînement: {X_train.shape}, Test: {X_test.shape}")
    
    # 2. Entraînement des modèles individuels
    logger.info("Étape 2: Entraînement des modèles individuels")
    model_factory = ModelFactory()
    
    # Modèles à entraîner (concepts du cours)
    models_config = {
        'linear_regression': {},
        'svm_linear': {'C': 1.0},
        'svm_rbf': {'C': 1.0, 'gamma': 'scale'},
        'decision_tree': {'max_depth': 10},
        'random_forest': {'n_estimators': 100, 'max_depth': 10},
        'neural_network': {'hidden_layer_sizes': (50, 25), 'alpha': 0.01}
    }
    
    trained_models = {}
    for model_name, params in models_config.items():
        logger.info(f"Entraînement du modèle: {model_name}")
        model = model_factory.create_model(model_name, params)
        model.fit(X_train, y_train)
        trained_models[model_name] = model
    
    # 3. Modèles ensemblistes
    logger.info("Étape 3: Modèles ensemblistes")
    ensemble_model = EnsembleModel()
    
    # Bagging (concept 9.2 du cours)
    bagging_model = ensemble_model.create_bagging_model()
    bagging_model.fit(X_train, y_train)
    trained_models['bagging'] = bagging_model
    
    # AdaBoost (concept 9.4 du cours)
    adaboost_model = ensemble_model.create_adaboost_model()
    adaboost_model.fit(X_train, y_train)
    trained_models['adaboost'] = adaboost_model
    
    # 4. Modèles probabilistes
    logger.info("Étape 4: Modèles probabilistes")
    probabilistic_forecaster = ProbabilisticForecaster()
    
    # GMM pour clustering des régimes de marché (concept 12.4)
    regime_model = probabilistic_forecaster.fit_market_regimes(data)
    
    # Modèle bayésien (concept 14.1)
    bayesian_predictions = probabilistic_forecaster.bayesian_inference(X_train, X_test, y_train)
    
    # 5. Évaluation des modèles
    logger.info("Étape 5: Évaluation des modèles")
    evaluator = ModelEvaluator()
    
    results = {}
    for name, model in trained_models.items():
        if name not in ['gmm', 'bayesian']:
            y_pred = model.predict(X_test)
            metrics = evaluator.calculate_regression_metrics(y_test, y_pred)
            results[name] = metrics
            logger.info(f"{name}: RMSE = {metrics['rmse']:.4f}, R² = {metrics['r2']:.4f}")
    
    # Évaluation des modèles probabilistes
    bayesian_metrics = evaluator.calculate_regression_metrics(y_test, bayesian_predictions)
    results['bayesian'] = bayesian_metrics
    
    # 6. Sélection du meilleur modèle et prédictions finales
    logger.info("Étape 6: Sélection du meilleur modèle")
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    best_model = trained_models.get(best_model_name)
    
    logger.info(f"Meilleur modèle: {best_model_name} avec RMSE: {results[best_model_name]['rmse']:.4f}")
    
    # Prédictions finales
    if best_model:
        final_predictions = best_model.predict(X_test)
        
        # Sauvegarde des résultats
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': final_predictions,
            'date': data.index[-len(y_test):]
        })
        
        predictions_df.to_csv('data/predictions.csv', index=False)
        logger.info("Prédictions sauvegardées dans data/predictions.csv")
    
    # 7. Visualisation des résultats
    logger.info("Étape 7: Génération des visualisations")
    evaluator.plot_predictions(y_test, final_predictions, 'Best Model Predictions')
    evaluator.plot_residuals(y_test, final_predictions)
    
    # Analyse des features importantes
    if hasattr(best_model, 'feature_importances_'):
        evaluator.plot_feature_importance(best_model, feature_names)
    
    logger.info("Projet terminé avec succès!")

if __name__ == "__main__":
    main()