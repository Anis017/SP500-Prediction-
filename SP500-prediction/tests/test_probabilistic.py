import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ajouter le chemin src au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from probabilistic import ProbabilisticForecaster
from data_prep import DataPreprocessor

class TestProbabilisticModels(unittest.TestCase):
    
    def setUp(self):
        """Setup pour les tests probabilistes"""
        self.prob_forecaster = ProbabilisticForecaster()
        
        # Génération de données financières simulées pour les tests
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Simulation de prix avec tendance et saisonnalité
        trend = np.linspace(100, 500, 1000)
        noise = np.random.normal(0, 10, 1000)
        seasonal = 50 * np.sin(2 * np.pi * np.arange(1000) / 365)
        
        self.test_data = pd.DataFrame({
            'Close': trend + seasonal + noise,
            'Volume': np.random.lognormal(10, 1, 1000),
            'interest_rate': np.random.normal(2.5, 0.5, 1000),
            'inflation': np.random.normal(2.0, 0.3, 1000),
            'vix': np.random.normal(15, 5, 1000)
        }, index=dates)
        
        # Ajout des returns et volatilité pour les tests
        self.test_data['returns'] = self.test_data['Close'].pct_change()
        self.test_data['volatility_20'] = self.test_data['returns'].rolling(20).std()
        self.test_data['rsi_14'] = self._calculate_test_rsi(self.test_data['Close'])
        
        # Nettoyage des NaN
        self.test_data = self.test_data.dropna()
        
        # Données d'entraînement/test
        self.X_train = np.random.randn(800, 5)
        self.X_test = np.random.randn(200, 5)
        self.y_train = np.random.randn(800)
        
    def _calculate_test_rsi(self, prices, window=14):
        """Calcul simplifié du RSI pour les tests"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def test_gmm_market_regimes(self):
        """Test de la détection des régimes de marché avec GMM"""
        print("Testing GMM Market Regimes...")
        
        # Test avec différents nombres de composants
        for n_components in [2, 3, 4]:
            with self.subTest(n_components=n_components):
                gmm_model = self.prob_forecaster.fit_market_regimes(
                    self.test_data, n_components=n_components
                )
                
                # Vérifications de base
                self.assertIsNotNone(gmm_model)
                self.assertEqual(gmm_model.n_components, n_components)
                self.assertTrue(hasattr(gmm_model, 'means_'))
                self.assertTrue(hasattr(gmm_model, 'covariances_'))
                
                # Vérifier que le modèle est bien entraîné
                self.assertTrue(gmm_model.converged_)
                
                print(f"✓ GMM with {n_components} components trained successfully")
    
    def test_bayesian_inference(self):
        """Test de l'inférence bayésienne"""
        print("Testing Bayesian Inference...")
        
        # Test de la régression bayésienne
        predictions = self.prob_forecaster.bayesian_inference(
            self.X_train, self.X_test, self.y_train
        )
        
        # Vérifications
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
        # Vérifier que le modèle est sauvegardé
        self.assertIn('bayesian', self.prob_forecaster.models)
        bayesian_model = self.prob_forecaster.models['bayesian']
        self.assertIsNotNone(bayesian_model)
        
        # Vérifier que le modèle a bien appris
        self.assertTrue(hasattr(bayesian_model, 'coef_'))
        
        print("✓ Bayesian inference completed successfully")
    
    def test_gaussian_process_regression(self):
        """Test de la régression par processus gaussiens"""
        print("Testing Gaussian Process Regression...")
        
        # Test GPR
        y_pred, y_std = self.prob_forecaster.gaussian_process_regression(
            self.X_train, self.X_test, self.y_train
        )
        
        # Vérifications de base
        self.assertIsNotNone(y_pred)
        self.assertIsNotNone(y_std)
        self.assertEqual(len(y_pred), len(self.X_test))
        self.assertEqual(len(y_std), len(self.X_test))
        
        # Vérifier que les incertitudes sont positives
        self.assertTrue(np.all(y_std >= 0))
        
        # Vérifier la cohérence des prédictions
        self.assertFalse(np.any(np.isnan(y_pred)))
        self.assertFalse(np.any(np.isnan(y_std)))
        
        print("✓ Gaussian Process Regression completed successfully")
    
    def test_monte_carlo_simulation(self):
        """Test des simulations Monte Carlo"""
        print("Testing Monte Carlo Simulations...")
        
        # Générer des returns simulés
        returns = self.test_data['returns'].dropna()
        
        # Tests avec différents paramètres
        test_cases = [
            (100, 10),   # 100 simulations, 10 jours
            (500, 30),   # 500 simulations, 30 jours
            (1000, 50)   # 1000 simulations, 50 jours
        ]
        
        for n_simulations, days in test_cases:
            with self.subTest(n_simulations=n_simulations, days=days):
                simulations = self.prob_forecaster.monte_carlo_simulation(
                    returns, n_simulations=n_simulations, days=days
                )
                
                # Vérifications
                self.assertIsNotNone(simulations)
                self.assertEqual(simulations.shape, (n_simulations, days))
                self.assertFalse(np.any(np.isnan(simulations)))
                
                # Vérifier que les simulations ont une variance raisonnable
                final_prices = simulations[:, -1]
                price_std = np.std(final_prices)
                self.assertGreater(price_std, 0)  # Doit avoir une certaine variabilité
                
                print(f"✓ Monte Carlo simulation ({n_simulations}, {days}) completed successfully")
    
    def test_value_at_risk(self):
        """Test du calcul de la Value at Risk"""
        print("Testing Value at Risk Calculation...")
        
        returns = self.test_data['returns'].dropna()
        
        # Test avec différents niveaux de confiance
        confidence_levels = [0.01, 0.05, 0.10]
        
        for confidence in confidence_levels:
            with self.subTest(confidence=confidence):
                var = self.prob_forecaster.calculate_value_at_risk(
                    returns, confidence_level=confidence
                )
                
                # Vérifications
                self.assertIsNotNone(var)
                self.assertIsInstance(var, (float, np.floating))
                
                # La VaR doit être négative (risque de perte)
                self.assertLess(var, 0)
                
                # Vérifier que la VaR est dans une plage raisonnable
                returns_min = returns.min()
                self.assertGreaterEqual(var, returns_min)
                
                print(f"✓ VaR at {confidence*100}% confidence level: {var:.4f}")
    
    def test_probabilistic_predictions_consistency(self):
        """Test de la cohérence entre différents modèles probabilistes"""
        print("Testing Probabilistic Models Consistency...")
        
        # Obtenir les prédictions de différents modèles
        bayesian_pred = self.prob_forecaster.bayesian_inference(
            self.X_train, self.X_test, self.y_train
        )
        
        gpr_pred, gpr_std = self.prob_forecaster.gaussian_process_regression(
            self.X_train, self.X_test, self.y_train
        )
        
        # Vérifier que les prédictions ont la même forme
        self.assertEqual(len(bayesian_pred), len(gpr_pred))
        
        # Vérifier que les prédictions ne sont pas identiques (différents modèles)
        correlation = np.corrcoef(bayesian_pred, gpr_pred)[0, 1]
        self.assertNotEqual(correlation, 1.0)  # Ne doivent pas être parfaitement corrélés
        
        # Vérifier que les écarts-types de GPR sont raisonnables
        self.assertTrue(np.all(gpr_std >= 0))
        avg_std = np.mean(gpr_std)
        self.assertGreater(avg_std, 0)  # Doit avoir une certaine incertitude
        
        print("✓ Probabilistic models show consistent but diverse predictions")
    
    def test_edge_cases(self):
        """Test des cas limites pour les modèles probabilistes"""
        print("Testing Edge Cases...")
        
        # Test avec très peu de données
        small_X = np.random.randn(5, 3)
        small_y = np.random.randn(5)
        
        # Bayesian inference devrait fonctionner même avec peu de données
        small_pred = self.prob_forecaster.bayesian_inference(
            small_X, small_X, small_y
        )
        self.assertEqual(len(small_pred), len(small_X))
        
        # Test avec données constantes
        constant_returns = pd.Series([0.01] * 100)  # Returns constants
        var_constant = self.prob_forecaster.calculate_value_at_risk(constant_returns)
        self.assertEqual(var_constant, 0.01)  # VaR devrait être égale au return constant
        
        # Test Monte Carlo avec très peu de simulations
        few_simulations = self.prob_forecaster.monte_carlo_simulation(
            self.test_data['returns'].dropna(), n_simulations=10, days=5
        )
        self.assertEqual(few_simulations.shape, (10, 5))
        
        print("✓ Edge cases handled successfully")
    
    def test_model_persistence(self):
        """Test de la persistance des modèles entraînés"""
        print("Testing Model Persistence...")
        
        # Entraîner plusieurs modèles
        gmm_model = self.prob_forecaster.fit_market_regimes(self.test_data)
        bayesian_pred = self.prob_forecaster.bayesian_inference(
            self.X_train, self.X_test, self.y_train
        )
        
        # Vérifier que les modèles sont bien sauvegardés
        self.assertIn('gmm', self.prob_forecaster.models)
        self.assertIn('bayesian', self.prob_forecaster.models)
        
        gmm_saved = self.prob_forecaster.models['gmm']
        bayesian_saved = self.prob_forecaster.models['bayesian']
        
        self.assertIsNotNone(gmm_saved)
        self.assertIsNotNone(bayesian_saved)
        
        # Vérifier que les modèles sauvegardés peuvent faire des prédictions
        if hasattr(gmm_saved, 'predict'):
            gmm_predictions = gmm_saved.predict(self.test_data[['returns', 'volatility_20', 'vix', 'rsi_14']].dropna())
            self.assertIsNotNone(gmm_predictions)
        
        bayesian_new_pred = bayesian_saved.predict(self.X_test)
        self.assertEqual(len(bayesian_new_pred), len(self.X_test))
        
        print("✓ Model persistence verified successfully")

    def test_probabilistic_metrics(self):
        """Test des métriques probabilistes spécifiques"""
        print("Testing Probabilistic Metrics...")
        
        # Test de la calibration des incertitudes
        _, gpr_std = self.prob_forecaster.gaussian_process_regression(
            self.X_train, self.X_test, self.y_train
        )
        
        # Vérifier que l'incertitude est corrélée avec l'erreur de prédiction
        gpr_pred, _ = self.prob_forecaster.gaussian_process_regression(
            self.X_train, self.X_test, self.y_train
        )
        
        # Pour un test significatif, nous aurions besoin de vraies valeurs cibles
        # Ici, nous vérifions simplement que l'incertitude est calculée
        uncertainty_mean = np.mean(gpr_std)
        self.assertGreater(uncertainty_mean, 0)
        
        # Test de la distribution des prédictions Monte Carlo
        returns = self.test_data['returns'].dropna()
        simulations = self.prob_forecaster.monte_carlo_simulation(
            returns, n_simulations=1000, days=10
        )
        
        # Vérifier que la distribution a les propriétés attendues
        final_returns = simulations[:, -1] - 1  # Convertir en returns
        mean_return = np.mean(final_returns)
        std_return = np.std(final_returns)
        
        # La moyenne devrait être proche du return historique moyen
        historical_mean = returns.mean()
        self.assertAlmostEqual(mean_return, historical_mean, delta=0.1)
        
        print("✓ Probabilistic metrics calculated correctly")

def run_probabilistic_tests():
    """Fonction pour exécuter tous les tests probabilistes"""
    print("=" * 60)
    print("EXÉCUTION DES TESTS PROBABILISTES")
    print("=" * 60)
    
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestProbabilisticModels)
    
    # Exécuter les tests avec une sortie verbose
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Afficher un résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS PROBABILISTES")
    print("=" * 60)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("🎉 TOUS LES TESTS PROBABILISTES ONT RÉUSSI !")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        for test, traceback in result.failures + result.errors:
            print(f"\nÉchec dans: {test}")
            print(f"Traceback: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Exécuter tous les tests
    success = run_probabilistic_tests()
    
    # Retourner un code de sortie approprié
    sys.exit(0 if success else 1)