import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

class ProbabilisticForecaster:
    """Modèles probabilistes (Partie IV du cours)"""
    
    def __init__(self):
        self.models = {}
    
    def fit_market_regimes(self, data, n_components=3):
        """Détection des régimes de marché avec GMM (concept 12.4)"""
        # Features pour la détection de régime
        regime_features = ['returns', 'volatility_20', 'vix', 'rsi_14']
        X_regime = data[regime_features].dropna()
        
        # Application du GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        regimes = gmm.fit_predict(X_regime)
        
        # Ajout des régimes aux données
        data_with_regimes = data.copy()
        data_with_regimes['market_regime'] = np.nan
        data_with_regimes.loc[X_regime.index, 'market_regime'] = regimes
        
        self.models['gmm'] = gmm
        return gmm
    
    def bayesian_inference(self, X_train, X_test, y_train):
        """Inférence bayésienne pour la prédiction (concept 14.1)"""
        # Régression bayésienne
        bayesian_model = BayesianRidge()
        bayesian_model.fit(X_train, y_train)
        
        # Prédictions avec intervalles de confiance
        y_pred, y_std = bayesian_model.predict(X_test, return_std=True)
        
        self.models['bayesian'] = bayesian_model
        return y_pred
    
    def gaussian_process_regression(self, X_train, X_test, y_train):
        """Régression par processus gaussiens"""
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, random_state=42)
        gp.fit(X_train, y_train)
        
        y_pred, y_std = gp.predict(X_test, return_std=True)
        return y_pred, y_std
    
    def monte_carlo_simulation(self, returns, n_simulations=1000, days=30):
        """Simulation Monte Carlo pour la prédiction de prix"""
        log_returns = np.log(1 + returns.dropna())
        
        mu = log_returns.mean()
        sigma = log_returns.std()
        
        simulations = np.zeros((n_simulations, days))
        
        for i in range(n_simulations):
            daily_returns = np.random.normal(mu, sigma, days)
            price_path = np.cumprod(1 + daily_returns)
            simulations[i] = price_path
        
        return simulations
    
    def calculate_value_at_risk(self, returns, confidence_level=0.05):
        """Calcul de la Value at Risk (VaR)"""
        var = np.percentile(returns.dropna(), confidence_level * 100)
        return var