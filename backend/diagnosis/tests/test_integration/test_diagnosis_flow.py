"""
Интеграционные тесты для основного потока диагностики приложения diagnosis.

Тестирует полные пользовательские сценарии и взаимодействие между компонентами системы.
"""

from unittest.mock import patch

import pytest
from django.urls import reverse


@pytest.mark.integration
def test_home_page_loads_correctly(client):
    """Тест что главная страница загружается и содержит ключевые элементы."""
    response = client.get(reverse("home"))

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Медицинский Диагностический Помощник" in content
    assert "симптом" in content.lower()


@pytest.mark.integration
def test_symptom_selection_to_prediction(client, common_symptoms):
    """Тест полного цикла: выбор симптомов -> отправка -> получение результатов."""
    with (
        patch("diagnosis.views.model") as mock_model,
        patch("diagnosis.views.symptoms_list", common_symptoms),
        patch("diagnosis.views.diseases_list", ["Грипп", "Простуда", "COVID-19"]),
        patch("diagnosis.views.get_disease_description") as mock_desc,
        patch("diagnosis.views.get_disease_treatment") as mock_treat,
    ):

        mock_model.predict_proba.return_value = [[0.8, 0.15, 0.05]]
        mock_model.classes_ = ["Грипп", "Простуда", "COVID-19"]
        mock_desc.return_value = "Тестовое описание"
        mock_treat.return_value = "Тестовое лечение"

        response = client.post(reverse("predict"), {"symptoms": common_symptoms})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Ошибка" not in content
    assert "error" not in content.lower()


@pytest.mark.integration
def test_results_page_contains_disease_cards(client, common_symptoms):
    """Тест что страница результатов содержит элементы интерфейса."""
    with (
        patch("diagnosis.views.model") as mock_model,
        patch("diagnosis.views.symptoms_list", common_symptoms),
        patch("diagnosis.views.diseases_list", ["Грипп", "Простуда", "COVID-19"]),
        patch("diagnosis.views.get_disease_description") as mock_desc,
        patch("diagnosis.views.get_disease_treatment") as mock_treat,
    ):

        mock_model.predict_proba.return_value = [[0.8, 0.15, 0.05]]
        mock_model.classes_ = ["Грипп", "Простуда", "COVID-19"]
        mock_desc.return_value = "Описание"
        mock_treat.return_value = "Лечение"

        response = client.post(reverse("predict"), {"symptoms": common_symptoms})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content
    assert "body" in content


@pytest.mark.integration
def test_navigation_from_results_to_disease_detail(client):
    """Тест навигации от страницы результатов к деталям заболевания."""
    with patch("diagnosis.views.get_disease_info") as mock_disease_info:
        mock_disease_info.return_value = {
            "description": "Тестовое описание гриппа",
            "treatment": "Тестовое лечение",
            "symptoms": ["температура", "кашель"],
            "severity": "medium",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        disease_response = client.get(reverse("disease_detail", args=["Грипп"]))

    assert disease_response.status_code == 200
    disease_content = disease_response.content.decode("utf-8")
    assert "не найдено" not in disease_content.lower()


@pytest.mark.integration
def test_empty_symptoms_submission(client):
    """Тест отправки формы без выбранных симптомов."""
    with patch("diagnosis.views.model"):
        response = client.post(reverse("predict"), {"symptoms": []})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content


@pytest.mark.integration
def test_single_symptom_diagnosis(client, minimal_symptoms):
    """Тест диагностики с одним симптомом."""
    with (
        patch("diagnosis.views.model") as mock_model,
        patch("diagnosis.views.symptoms_list", minimal_symptoms),
        patch("diagnosis.views.diseases_list", ["Синдром хронической усталости"]),
        patch("diagnosis.views.get_disease_description") as mock_desc,
        patch("diagnosis.views.get_disease_treatment") as mock_treat,
    ):

        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = ["Синдром хронической усталости"]
        mock_desc.return_value = "Описание"
        mock_treat.return_value = "Лечение"

        response = client.post(reverse("predict"), {"symptoms": minimal_symptoms})

    assert response.status_code == 200


@pytest.mark.integration
def test_multiple_symptoms_diagnosis(client, respiratory_symptoms):
    """Тест диагностики с множеством симптомов."""
    with (
        patch("diagnosis.views.model") as mock_model,
        patch("diagnosis.views.symptoms_list", respiratory_symptoms),
        patch("diagnosis.views.diseases_list", ["ОРВИ"]),
        patch("diagnosis.views.get_disease_description") as mock_desc,
        patch("diagnosis.views.get_disease_treatment") as mock_treat,
    ):

        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = ["ОРВИ"]
        mock_desc.return_value = "Описание"
        mock_treat.return_value = "Лечение"

        response = client.post(reverse("predict"), {"symptoms": respiratory_symptoms})

    assert response.status_code == 200


@pytest.mark.integration
def test_complete_user_journey(client, common_symptoms):
    """Полный тест пользовательского сценария от начала до конца."""
    # 1. Главная страница
    home_response = client.get(reverse("home"))
    assert home_response.status_code == 200

    # 2. Диагностика с симптомами
    with (
        patch("diagnosis.views.model") as mock_model,
        patch("diagnosis.views.symptoms_list", common_symptoms),
        patch("diagnosis.views.diseases_list", ["Грипп"]),
        patch("diagnosis.views.get_disease_description") as mock_desc,
        patch("diagnosis.views.get_disease_treatment") as mock_treat,
    ):

        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = ["Грипп"]
        mock_desc.return_value = "Описание"
        mock_treat.return_value = "Лечение"

        predict_response = client.post(reverse("predict"), {"symptoms": common_symptoms})
        assert predict_response.status_code == 200

    # 3. Детали заболевания
    with patch("diagnosis.views.get_disease_info") as mock_disease_info:
        mock_disease_info.return_value = {
            "description": "Описание гриппа",
            "treatment": "Лечение гриппа",
            "symptoms": ["температура", "кашель"],
            "severity": "medium",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        detail_response = client.get(reverse("disease_detail", args=["Грипп"]))
        assert detail_response.status_code == 200

    # 4. Возврат к новой диагностике
    new_diagnosis_response = client.get(reverse("home"))
    assert new_diagnosis_response.status_code == 200
