import unittest
import numpy as np
import pandas as pd
from src.models import ModelFactory, AdvancedModels
from src.data_prep import DataPreprocessor

class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Setup pour les tests"""
        self.model_factory = ModelFactory()
        self.advanced_models = AdvancedModels()
        
        # Données de test
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.randn(100)
        self.X_test = np.random.randn(20, 10)
        
    def test_model_creation(self):
        """Test de création des modèles"""
        models_to_test = ['linear_regression', 'svm_linear', 'decision_tree']
        
        for model_type in models_to_test:
            model = self.model_factory.create_model(model_type)
            self.assertIsNotNone(model)
            
    def test_knn_regressor(self):
        """Test du K-NN Regressor"""
        knn = self.advanced_models.create_knn_regressor(k=3)
        knn.fit(self.X_train, self.y_train)
        predictions = knn.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        
    def test_model_complexity_evaluation(self):
        """Test de l'évaluation de la complexité"""
        model = self.model_factory.create_model('decision_tree')
        model.fit(self.X_train, self.y_train)
        
        complexity_info = self.advanced_models.evaluate_model_complexity(
            model, self.X_train, self.X_test, self.y_train, self.y_train[:len(self.X_test)]
        )
        
        self.assertIn('train_mse', complexity_info)
        self.assertIn('test_mse', complexity_info)
        self.assertIn('overfitting', complexity_info)

if __name__ == '__main__':
    unittest.main()