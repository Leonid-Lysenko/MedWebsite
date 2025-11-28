"""
Интеграционные тесты для навигации по приложению diagnosis.

Проверяет доступность страниц, корректность навигационных ссылок и переходы между страницами.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from django.urls import reverse


@pytest.mark.integration
def test_all_main_pages_accessible(client, navigation_urls):
    """Тест что все основные страницы приложения доступны."""
    static_pages = ["home", "about", "how_to_use", "knowledge_base"]

    for page in static_pages:
        url = navigation_urls[page]
        response = client.get(url)
        assert response.status_code == 200, f"Страница {page} недоступна"

        content = response.content.decode("utf-8")
        assert "html" in content
        assert "body" in content


@pytest.mark.integration
def test_navigation_menu_links(client):
    """Тест что навигационное меню содержит все основные ссылки."""
    response = client.get(reverse("home"))
    content = response.content.decode("utf-8")

    nav_links = ["Главная", "О нас", "Как пользоваться", "База знаний"]

    for link in nav_links:
        assert link in content, f"Ссылка '{link}' отсутствует в навигации"


@pytest.mark.integration
def test_breadcrumb_navigation_from_knowledge_base(client):
    """Тест хлебных крошек при переходе из базы знаний."""
    mock_model = Mock()
    mock_model.classes_ = np.array(["Грипп"])

    with (
        patch("diagnosis.views.model", mock_model),
        patch("diagnosis.views.get_disease_info") as mock_disease_info,
    ):

        mock_disease_info.return_value = {
            "description": "Описание гриппа",
            "treatment": "Лечение гриппа",
            "symptoms": ["температура"],
            "severity": "high",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        response = client.get("/knowledge-base/disease/Грипп/")

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Главная" in content or "Главная" in content
    assert "База знаний" in content


@pytest.mark.integration
def test_breadcrumb_navigation_from_results(client):
    """Тест хлебных крошек при переходе из результатов диагностики."""
    with patch("diagnosis.views.get_disease_info") as mock_disease_info:
        mock_disease_info.return_value = {
            "description": "Описание гриппа",
            "treatment": "Лечение гриппа",
            "symptoms": ["температура"],
            "severity": "high",
            "specialist": "Терапевт",
            "category": "Инфекционные",
        }

        response = client.get("/disease/Грипп/")

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Главная" in content or "Главная" in content
    assert "Результаты диагностики" in content or "Результаты" in content


@pytest.mark.integration
def test_return_to_home_from_any_page(client):
    """Тест что с любой страницы можно вернуться на главную."""
    pages_to_test = [reverse("about"), reverse("how_to_use"), reverse("knowledge_base")]

    for page_url in pages_to_test:
        response = client.get(page_url)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        assert "Главная" in content or "home" in content.lower() or "/" in content


@pytest.mark.integration
def test_disease_detail_accessibility(client, common_disease_names):
    """Тест доступности страниц деталей заболеваний."""
    for disease_name in common_disease_names:
        with patch("diagnosis.views.get_disease_info") as mock_disease_info:
            mock_disease_info.return_value = {
                "description": f"Описание {disease_name}",
                "treatment": f"Лечение {disease_name}",
                "symptoms": ["симптом1"],
                "severity": "medium",
                "specialist": "Терапевт",
                "category": "Инфекционные",
            }

            response = client.get(reverse("disease_detail", args=[disease_name]))
            assert response.status_code == 200
            content = response.content.decode("utf-8")
            assert "html" in content


@pytest.mark.integration
def test_knowledge_base_disease_detail_accessibility(client, common_disease_names):
    """Тест доступности страниц деталей заболеваний из базы знаний."""
    for disease_name in common_disease_names:
        with patch("diagnosis.views.get_disease_info") as mock_disease_info:
            mock_disease_info.return_value = {
                "description": f"Описание {disease_name}",
                "treatment": f"Лечение {disease_name}",
                "symptoms": ["симптом1"],
                "severity": "medium",
                "specialist": "Терапевт",
                "category": "Инфекционные",
            }

            response = client.get("/knowledge-base/disease/{}/".format(disease_name))
            assert response.status_code == 200


@pytest.mark.integration
def test_error_pages_handling(client):
    """Тест обработки ошибок и недоступных страниц."""
    invalid_urls = [
        "/nonexistent-page/",
        "/disease/NonExistentDisease123/",
        "/invalid-path/",
    ]

    for url in invalid_urls:
        response = client.get(url)
        assert response.status_code != 500

        if response.status_code == 404:
            pass
        else:
            content = response.content.decode("utf-8")
            assert "html" in content


@pytest.mark.integration
def test_complete_navigation_flow(client):
    """Полный тест навигационного потока через все приложение."""
    response = client.get(reverse("home"))
    assert response.status_code == 200

    response = client.get(reverse("about"))
    assert response.status_code == 200

    response = client.get(reverse("how_to_use"))
    assert response.status_code == 200

    mock_model = Mock()
    mock_model.classes_ = np.array(["Грипп"])

    with patch("diagnosis.views.model", mock_model):
        response = client.get(reverse("knowledge_base"))
        assert response.status_code == 200

    response = client.get(reverse("home"))
    assert response.status_code == 200
