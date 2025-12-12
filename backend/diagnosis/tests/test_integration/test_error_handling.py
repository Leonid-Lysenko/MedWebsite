"""
Интеграционные тесты для обработки ошибок приложения diagnosis.

Тестирует устойчивость системы к некорректным данным.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from django.urls import reverse


@pytest.mark.integration
def test_predict_with_invalid_symptoms(client, error_scenarios):
    """Тест обработки невалидных симптомов в предсказании."""
    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["Грипп"])

        response = client.post(reverse("predict"), {"symptoms": error_scenarios["invalid_symptoms"]})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content


@pytest.mark.integration
def test_predict_with_special_characters(client, error_scenarios):
    """Тест обработки симптомов со специальными символами."""
    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["Грипп"])

        response = client.post(reverse("predict"), {"symptoms": error_scenarios["special_chars"]})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content
    assert "body" in content
    assert "Internal Server Error" not in content


@pytest.mark.integration
def test_predict_with_empty_symptoms(client, error_scenarios):
    """Тест обработки пустого списка симптомов."""
    with patch("diagnosis.views.model"):
        response = client.post(reverse("predict"), {"symptoms": error_scenarios["empty_data"]})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content


@pytest.mark.integration
def test_disease_detail_nonexistent_disease(client, nonexistent_diseases):
    """Тест запроса несуществующих заболеваний."""
    for disease_name in nonexistent_diseases:
        response = client.get(reverse("disease_detail", args=[disease_name]))

        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "html" in content
        assert "Internal Server Error" not in content


@pytest.mark.integration
def test_knowledge_base_nonexistent_disease(client, nonexistent_diseases):
    """Тест запроса несуществующих заболеваний через базу знаний."""
    for disease_name in nonexistent_diseases:
        response = client.get("/knowledge-base/disease/{}/".format(disease_name))

        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "html" in content
        assert "Internal Server Error" not in content


@pytest.mark.integration
def test_invalid_urls_handling(client, invalid_urls):
    """Тест обработки некорректных URL."""
    for url in invalid_urls:
        response = client.get(url)

        assert response.status_code != 500

        if response.status_code == 404:
            pass
        else:
            content = response.content.decode("utf-8")
            assert "html" in content


@pytest.mark.integration
def test_ml_model_failure_handling(client, common_symptoms):
    """Тест обработки сбоя ML модели."""
    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.side_effect = Exception("ML model failed")

        response = client.post(reverse("predict"), {"symptoms": common_symptoms})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content
    assert "Internal Server Error" not in content


@pytest.mark.integration
def test_large_input_handling(client):
    """Тест обработки большого количества симптомов."""
    large_symptoms_list = ["Симптом{}".format(i) for i in range(100)]

    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["Грипп"])

        response = client.post(reverse("predict"), {"symptoms": large_symptoms_list})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content


@pytest.mark.integration
def test_unicode_handling(client):
    """Тест обработки Unicode символов."""
    unicode_symptoms = ["Головная боль 😫", "Температура 🌡️", "Кашель 🤧"]

    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["Грипп"])

        response = client.post(reverse("predict"), {"symptoms": unicode_symptoms})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content


@pytest.mark.integration
def test_error_page_display(client):
    """Тест что страница ошибки корректно отображается."""
    with patch("diagnosis.views.get_disease_suggestions") as mock_suggestions:
        mock_suggestions.return_value = ["Грипп", "Простуда"]

        response = client.get(reverse("disease_detail", args=["НесуществующаяБолезнь"]))

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "не найдено" in content.lower() or "not found" in content.lower()
    assert "html" in content
    assert "body" in content
