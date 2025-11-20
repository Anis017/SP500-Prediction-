import numpy as np
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class EnsembleModel:
    """Implémentation des méthodes ensemblistes (Partie III du cours)"""
    
    def __init__(self):
        self.models = {}
    
    def create_bagging_model(self, n_estimators=50, max_samples=0.8, random_state=42):
        """Bagging Regressor (concept 9.2)"""
        base_estimator = DecisionTreeRegressor(max_depth=10, random_state=random_state)
        return BaggingRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
    
    def create_adaboost_model(self, n_estimators=50, learning_rate=0.1, random_state=42):
        """AdaBoost Regressor (concept 9.4)"""
        base_estimator = DecisionTreeRegressor(max_depth=5, random_state=random_state)
        return AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
    
    def create_random_forest_model(self, n_estimators=100, max_depth=15, random_state=42):
        """Random Forest (concept 9.3)"""
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def create_stacking_ensemble(self, base_models, meta_model):
        """Stacking ensemble personnalisé"""
        from sklearn.base import clone
        
        class StackingEnsemble:
            def __init__(self, base_models, meta_model):
                self.base_models = [clone(model) for model in base_models]
                self.meta_model = clone(meta_model)
                
            def fit(self, X, y):
                # Entraînement des modèles de base
                for model in self.base_models:
                    model.fit(X, y)
                
                # Prédictions pour le meta-modèle
                base_predictions = np.column_stack([
                    model.predict(X) for model in self.base_models
                ])
                
                # Entraînement du meta-modèle
                self.meta_model.fit(base_predictions, y)
                return self
                
            def predict(self, X):
                base_predictions = np.column_stack([
                    model.predict(X) for model in self.base_models
                ])
                return self.meta_model.predict(base_predictions)
        
        return StackingEnsemble(base_models, meta_model)
    
    def evaluate_ensemble_diversity(self, models, X, y):
        """Évaluation de la diversité des modèles dans l'ensemble"""
        predictions = np.array([model.predict(X) for model in models])
        
        # Calcul de la corrélation moyenne entre les prédictions
        correlation_matrix = np.corrcoef(predictions)
        np.fill_diagonal(correlation_matrix, 0)  # Ignorer l'auto-corrélation
        
        average_correlation = np.mean(np.abs(correlation_matrix))
        diversity = 1 - average_correlation
        
        return {
            'average_correlation': average_correlation,
            'diversity': diversity,
            'high_diversity': diversity > 0.5
        }