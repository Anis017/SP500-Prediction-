import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Évaluation avancée des modèles selon les concepts du cours"""
    
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_regression_metrics(self, y_true, y_pred):
        """Calcul des métriques de régression"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': self._mean_absolute_percentage_error(y_true, y_pred)
        }
        return metrics
    
    def _mean_absolute_percentage_error(self, y_true, y_pred):
        """Calcul du MAPE"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def plot_predictions(self, y_true, y_pred, title="Prédictions vs Réalité"):
        """Visualisation des prédictions"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title(f'{title} - Scatter Plot')
        
        plt.subplot(1, 2, 2)
        plt.plot(y_true.values, label='Réel', alpha=0.7)
        plt.plot(y_pred, label='Prédit', alpha=0.7)
        plt.xlabel('Temps')
        plt.ylabel('Rendement')
        plt.title(f'{title} - Séries Temporelles')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/model_performance/predictions_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_residuals(self, y_true, y_pred):
        """Analyse des résidus"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Prédictions')
        plt.ylabel('Résidus')
        plt.title('Résidus vs Prédictions')
        
        plt.subplot(1, 3, 2)
        plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Résidus')
        plt.ylabel('Fréquence')
        plt.title('Distribution des Résidus')
        
        plt.subplot(1, 3, 3)
        pd.Series(residuals).plot(kind='kde')
        plt.xlabel('Résidus')
        plt.ylabel('Densité')
        plt.title('Densité des Résidus')
        
        plt.tight_layout()
        plt.savefig('outputs/model_performance/residuals_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Visualisation de l'importance des features"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.title("Importance des Features")
            plt.bar(range(min(top_n, len(importances))), 
                   importances[indices[:top_n]])
            plt.xticks(range(min(top_n, len(importances))), 
                      [feature_names[i] for i in indices[:top_n]], rotation=45)
            plt.tight_layout()
            plt.savefig('outputs/feature_analysis/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_learning_curve(self, model, X, y, title="Courbe d'Apprentissage"):
        """Courbe d'apprentissage (concept 5.1)"""
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Erreur d'entraînement")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Erreur de validation")
        plt.xlabel("Taille de l'ensemble d'entraînement")
        plt.ylabel("MSE")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        plt.savefig('outputs/model_performance/learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()