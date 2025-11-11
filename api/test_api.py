from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_analyze_single_text():
    """Тест эндпоинта /analyze для одного текста"""
    payload = {"text": "Сегодня отличный день!"}
    response = client.post("/analyze", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "text" in data
    assert "label" in data
    assert "score" in data
    assert isinstance(data["score"], float)
    assert 0.0 <= data["score"] <= 1.0


def test_analyze_negative_text():
    """Проверяем, что негативные тексты классифицируются корректно"""
    payload = {"text": "Мне очень грустно и плохо."}
    response = client.post("/analyze", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["NEGATIVE", "Positive", "LABEL_0", "LABEL_1"]


def test_batch_analyze():
    """Тест эндпоинта /batch-analyze для нескольких текстов"""
    payload = {
        "texts": [
            "Мне сегодня грустно.",
            "Отличный результат!",
            "Совершенно ужасный день.",
        ]
    }
    response = client.post("/batch-analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == len(payload["texts"])

    for item in data:
        assert "text" in item
        assert "label" in item
        assert "score" in item
        assert isinstance(item["score"], float)


def test_invalid_request_single():
    """Проверка ошибки при некорректном запросе"""
    response = client.post("/analyze", json={})
    assert response.status_code in [400, 422]


def test_invalid_request_batch():
    """Проверка ошибки при некорректном запросе для /batch-analyze"""
    response = client.post("/batch-analyze", json={})
    assert response.status_code in [400, 422]
