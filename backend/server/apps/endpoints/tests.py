from django.test import TestCase
from rest_framework.test import APIClient

class EndpointTests(TestCase):

    def test_predict_view(self):
        client = APIClient()
        input_data = {
            "радость": 2,
            "благодарность": 3,
            "восторг": 0,
            "сильно": 1,
            "любить": 1,
            "мультфильм": 1,
            "фердинанд": 1,
            "бык": 1,
            "огромный": 1,
            "милый": 1
        }
        classifier_url = "/api/v1/sentiment_classifier/predict"
        response = client.post(classifier_url, input_data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["label"], "mid")
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)