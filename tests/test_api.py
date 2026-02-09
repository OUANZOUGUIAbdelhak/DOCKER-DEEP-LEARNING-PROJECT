"""
Unit tests for API endpoints
Optional test file for API validation
"""

import unittest
import requests
import json


class TestAPI(unittest.TestCase):
    """Test cases for REST API"""
    
    BASE_URL = "http://localhost:5000"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = requests.get(f"{self.BASE_URL}/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('architecture', data)
        self.assertIn('input_shape', data)
    
    def test_predict_endpoint(self):
        """Test prediction endpoint"""
        # Create dummy input (224x224x3 image flattened)
        dummy_input = [[0.5] * (224 * 224 * 3)]
        
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json={"data": dummy_input}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('predictions', data)
        self.assertIn('confidence', data)


if __name__ == '__main__':
    unittest.main()
