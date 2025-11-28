"""
Интеграционные тесты для базы знаний заболеваний приложения diagnosis.

Тестирует функциональность базы знаний, включая навигацию, поиск и отображение заболеваний.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from django.urls import reverse


@pytest.mark.integration
def test_knowledge_base_page_loads(client):
    """Тест что страница базы знаний загружается корректно."""
    mock_model = Mock()
    mock_model.classes_ = np.array(["Грипп", "Простуда"])

    with patch("diagnosis.views.model", mock_model):
        response = client.get(reverse("knowledge_base"))

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "База знаний заболеваний" in content


@pytest.mark.integration
def test_knowledge_base_contains_alphabet_navigation(client):
    """Тест что база знаний содержит алфавитную навигацию."""
    mock_model = Mock()
    mock_model.classes_ = np.array(["Ангина", "Бронхит", "Грипп"])

    with patch("diagnosis.views.model", mock_model):
        response = client.get(reverse("knowledge_base"))

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "А" in content or "Б" in content or "В" in content


@pytest.mark.integration
def test_knowledge_base_displays_disease_cards(client):
    """Тест что база знаний отображает карточки заболеваний."""
    mock_model = Mock()
    mock_model.classes_ = np.array(["Грипп", "Простуда"])

    with (
        patch("diagnosis.views.model", mock_model),
        patch("diagnosis.views.get_disease_info") as mock_disease_info,
    ):

        mock_disease_info.return_value = {
            "description": "Тестовое описание",
            "treatment": "Тестовое лечение",
            "symptoms": ["симптом1"],
            "severity": "medium",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        response = client.get(reverse("knowledge_base"))

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Грипп" in content or "Простуда" in content


@pytest.mark.integration
def test_disease_detail_from_knowledge_base(client):
    """Тест перехода из базы знаний к деталям заболевания."""
    with patch("diagnosis.views.get_disease_info") as mock_disease_info:
        mock_disease_info.return_value = {
            "description": "Острое инфекционное заболевание",
            "treatment": "Постельный режим, обильное питье",
            "symptoms": ["Высокая температура", "Кашель"],
            "severity": "high",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        response = client.get("/knowledge-base/disease/Грипп/")

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "не найдено" not in content.lower()


@pytest.mark.integration
def test_search_functionality_in_knowledge_base(client):
    """Тест поиска заболеваний в базе знаний."""
    mock_model = Mock()
    mock_model.classes_ = np.array(["Грипп", "Грипп A", "Простуда"])

    with patch("diagnosis.views.model", mock_model):
        response = client.get(reverse("knowledge_base"))

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "search" in content.lower() or "поиск" in content.lower()


@pytest.mark.integration
def test_severity_badges_display(client):
    """Тест что бейджи серьезности отображаются в базе знаний."""
    mock_model = Mock()
    mock_model.classes_ = np.array(["Грипп"])

    with (
        patch("diagnosis.views.model", mock_model),
        patch("diagnosis.views.get_disease_info") as mock_disease_info,
    ):

        mock_disease_info.return_value = {
            "description": "Описание",
            "treatment": "Лечение",
            "symptoms": ["симптом1"],
            "severity": "high",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        response = client.get(reverse("knowledge_base"))

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "badge" in content or "ВЫСОКАЯ" in content or "high" in content.lower()


@pytest.mark.integration
def test_category_display_in_knowledge_base(client):
    """Тест что категории заболеваний отображаются."""
    mock_model = Mock()
    mock_model.classes_ = np.array(["Грипп"])

    with (
        patch("diagnosis.views.model", mock_model),
        patch("diagnosis.views.get_disease_info") as mock_disease_info,
    ):

        mock_disease_info.return_value = {
            "description": "Описание",
            "treatment": "Лечение",
            "symptoms": ["симптом1"],
            "severity": "medium",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        response = client.get(reverse("knowledge_base"))

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Инфекционные" in content or "категория" in content.lower()


@pytest.mark.integration
def test_knowledge_base_with_empty_diseases(client):
    """Тест базы знаний когда нет заболеваний."""
    mock_model = Mock()
    mock_model.classes_ = np.array([])

    with patch("diagnosis.views.model", mock_model):
        response = client.get(reverse("knowledge_base"))

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content


@pytest.mark.integration
def test_complete_knowledge_base_journey(client):
    """Полный тест сценария работы с базой знаний."""
    home_response = client.get(reverse("home"))
    assert home_response.status_code == 200

    mock_model = Mock()
    mock_model.classes_ = np.array(["Грипп", "Простуда"])

    with (
        patch("diagnosis.views.model", mock_model),
        patch("diagnosis.views.get_disease_info") as mock_disease_info,
    ):

        mock_disease_info.return_value = {
            "description": "Тестовое описание",
            "treatment": "Тестовое лечение",
            "symptoms": ["симптом1"],
            "severity": "medium",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        kb_response = client.get(reverse("knowledge_base"))
        assert kb_response.status_code == 200

    with patch("diagnosis.views.get_disease_info") as mock_disease_info:
        mock_disease_info.return_value = {
            "description": "Детальное описание гриппа",
            "treatment": "Лечение гриппа",
            "symptoms": ["температура", "кашель"],
            "severity": "high",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        detail_response = client.get("/knowledge-base/disease/Грипп/")
        assert detail_response.status_code == 200

    kb_return_response = client.get(reverse("knowledge_base"))
    assert kb_return_response.status_code == 200
