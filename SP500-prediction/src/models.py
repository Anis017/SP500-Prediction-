import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ModelFactory:
    """Factory pour créer différents modèles de ML selon les concepts du cours"""
    
    def create_model(self, model_type, params=None):
        """Crée un modèle spécifique basé sur le type"""
        if params is None:
            params = {}
            
        models = {
            'linear_regression': LinearRegression(**params),
            'ridge': Ridge(**params),
            'lasso': Lasso(**params),
            'svm_linear': SVR(kernel='linear', **params),
            'svm_rbf': SVR(kernel='rbf', **params),
            'decision_tree': DecisionTreeRegressor(random_state=42, **params),
            'random_forest': RandomForestRegressor(random_state=42, **params),
            'neural_network': MLPRegressor(random_state=42, **params)
        }
        
        if model_type not in models:
            raise ValueError(f"Modèle non supporté: {model_type}")
            
        return models[model_type]
    
    def tune_hyperparameters(self, model, X_train, y_train, param_grid):
        """Optimisation des hyperparamètres (concept 5.3.4)"""
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_

class AdvancedModels:
    """Implémentation de modèles avancés du cours"""
    
    def __init__(self):
        self.models = {}
    
    def create_knn_regressor(self, k=5):
        """K-NN Regressor (concept 4.1)"""
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=k)
    
    def create_ensemble_voting(self, estimators):
        """Méthode d'ensemble par vote (concept 9.1)"""
        from sklearn.ensemble import VotingRegressor
        return VotingRegressor(estimators=estimators)
    
    def evaluate_model_complexity(self, model, X_train, X_test, y_train, y_test):
        """Évaluation de la complexité du modèle (concept 5.1)"""
        from sklearn.metrics import mean_squared_error
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        complexity_ratio = test_mse / train_mse if train_mse > 0 else float('inf')
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'complexity_ratio': complexity_ratio,
            'overfitting': complexity_ratio > 1.5
        }