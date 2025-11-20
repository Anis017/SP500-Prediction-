import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Classe pour la préparation des données du S&P 500 avec feature engineering avancé"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        
    def load_and_prepare_data(self, start_date='2010-01-01', end_date='2023-12-31'):
        """Chargement et préparation initiale des données"""
        # Téléchargement des données S&P 500
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)
        
        # Données économiques (simulées pour l'exemple)
        economic_data = self._generate_economic_data(sp500.index)
        
        # Fusion des données
        data = pd.merge(sp500, economic_data, left_index=True, right_index=True, how='left')
        
        # Nettoyage des données
        data = self._clean_data(data)
        
        return data
    
    def _generate_economic_data(self, dates):
        """Génération de données économiques simulées"""
        np.random.seed(42)
        n_points = len(dates)
        
        economic_data = pd.DataFrame(index=dates)
        economic_data['interest_rate'] = np.random.normal(2.5, 0.5, n_points)
        economic_data['inflation'] = np.random.normal(2.0, 0.3, n_points)
        economic_data['gdp_growth'] = np.random.normal(2.0, 0.8, n_points)
        economic_data['unemployment'] = np.random.normal(5.5, 1.0, n_points)
        economic_data['vix'] = np.random.normal(15, 5, n_points)  # Volatility Index
        
        return economic_data
    
    def _clean_data(self, data):
        """Nettoyage et imputation des données manquantes"""
        # Suppression des lignes avec trop de valeurs manquantes
        data = data.dropna(thresh=len(data.columns) - 5)
        
        # Imputation des valeurs manquantes
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
        data[numeric_columns] = data[numeric_columns].fillna(method='bfill')
        
        return data
    
    def create_advanced_features(self, data):
        """Création de features avancées selon les concepts du cours"""
        
        # Features de prix techniques
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Moyennes mobiles (concept de lissage)
        for window in [5, 10, 20, 50]:
            data[f'sma_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'ema_{window}'] = data['Close'].ewm(span=window).mean()
            data[f'price_vs_sma_{window}'] = data['Close'] / data[f'sma_{window}'] - 1
        
        # Volatilité (concept d'estimation de densité)
        for window in [5, 10, 20]:
            data[f'volatility_{window}'] = data['returns'].rolling(window=window).std()
        
        # RSI (Relative Strength Index)
        data['rsi_14'] = self._calculate_rsi(data['Close'], 14)
        
        # MACD (Moving Average Convergence Divergence)
        data['macd'], data['macd_signal'] = self._calculate_macd(data['Close'])
        
        # Features de momentum
        data['momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
        data['momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
        
        # Features de volume
        data['volume_sma_5'] = data['Volume'].rolling(5).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma_5']
        
        # Features économiques avancées
        data['real_interest_rate'] = data['interest_rate'] - data['inflation']
        data['economic_growth_risk'] = data['gdp_growth'] * data['vix'] / 100
        
        # Features de corrélation temporelle
        data['autocorr_5'] = data['returns'].rolling(5).apply(lambda x: x.autocorr(), raw=False)
        
        # Target variable: Rendement futur (5 jours)
        data['future_return_5d'] = data['Close'].shift(-5) / data['Close'] - 1
        
        # Suppression des lignes avec des valeurs manquantes
        data = data.dropna()
        
        return data
    
    def _calculate_rsi(self, prices, window=14):
        """Calcul du RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcul du MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def prepare_features_target(self, data, test_size=0.2):
        """Préparation des features et de la target pour l'apprentissage"""
        
        # Sélection des features
        feature_columns = [col for col in data.columns if col not in [
            'future_return_5d', 'Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close'
        ]]
        
        X = data[feature_columns]
        y = data['future_return_5d']
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Réduction de dimension avec PCA (concept 13)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Séparation train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=test_size, shuffle=False, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, feature_columns