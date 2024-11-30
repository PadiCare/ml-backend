import unittest
import json
from app import app  # Import your Flask app

class MLBackendTestCase(unittest.TestCase):
    def setUp(self):
        # Set up the Flask test client
        self.app = app.test_client()
        self.app.testing = True

    def test_home_endpoint(self):
        """Test the home endpoint."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)

    def test_predict_endpoint_with_valid_image(self):
        """Test the predict endpoint with a valid image from storage."""
        # Replace with a valid image ID that you know exists in your GCS bucket
        image_id = "200027.jpg"  # Update with an actual valid image ID
        response = self.app.post(
            '/predict',
            data=json.dumps({"imageId": image_id}),
            content_type='application/json'
        )
        print(response.data)  # Print response data for debugging
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('label', data)

    def test_predict_endpoint_with_invalid_image(self):
        """Test the predict endpoint with an invalid image ID."""
        image_id = "200027.jpg"  # Use a non-existent image ID
        response = self.app.post(
            '/predict',
            data=json.dumps({"imageId": image_id}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()