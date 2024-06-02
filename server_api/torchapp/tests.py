from django.test import TestCase, Client
import json


class TestClassifyText(TestCase):
    def setUp(self):
        self.client = Client()

    def test_classify_text_success(self):
        # Подготовка данных для запроса
        data = {"text": "Some text to classify"}

        # Выполнение запроса
        response = self.client.post(
            "/api/classify/", json.dumps(data), content_type="application/json"
        )

        # Проверка статуса кода ответа
        self.assertEqual(response.status_code, 200)

        # Проверка наличия ключей в ответе
        self.assertIn("text", response.json())
        self.assertIn("probabilities", response.json())

        # Проверка типа данных в ответе
        self.assertIsInstance(response.json()["text"], str)
        self.assertIsInstance(response.json()["probabilities"], list)

        # Проверка длины списка вероятностей
        self.assertEqual(len(response.json()["probabilities"]), 1)


class TestClassifyTextInvalidMethod(TestCase):
    def setUp(self):
        self.client = Client()

    def test_classify_text_invalid_method(self):
        # Выполнение GET запроса
        response = self.client.get("/api/classify/")

        # Проверка статуса кода ответа
        self.assertEqual(response.status_code, 405)


class TestClassifyTextError(TestCase):
    def setUp(self):
        self.client = Client()

    def test_classify_text_error(self):
        # Подготовка данных для запроса (без передачи текста)
        data = {}

        # Выполнение запроса
        response = self.client.post(
            "/api/classify/", json.dumps(data), content_type="application/json"
        )

        # Проверка статуса кода ответа
        self.assertEqual(response.status_code, 200)

        # Проверка содержимого ответа
        self.assertIn("text", response.json())
        self.assertIn("probabilities", response.json())
        self.assertEqual(response.json()["text"], "")
        self.assertEqual(response.json()["probabilities"], [1])
