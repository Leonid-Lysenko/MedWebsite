"""
Фикстуры для интеграционных тестов приложения diagnosis.

Содержит набор фикстур для тестирования интеграционных сценариев и потоков данных.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from django.test import Client


@pytest.fixture
def client():
    """Фикстура Django test client для тестирования HTTP-запросов."""
    return Client()


@pytest.fixture
def common_symptoms():
    """Фикстура распространенных симптомов для тестирования диагностики."""
    return ["Кашель", "Высокая температура", "Головная боль"]


@pytest.fixture
def respiratory_symptoms():
    """Фикстура симптомов респираторных заболеваний для специализированного тестирования."""
    return ["Кашель", "Насморк", "Боль в горле", "Чихание"]


@pytest.fixture
def minimal_symptoms():
    """Фикстура минимального набора симптомов для тестирования граничных случаев."""
    return ["Усталость"]


@pytest.fixture
def navigation_urls():
    """Фикстура всех основных URL приложения для тестирования навигации."""
    return {
        "home": "/",
        "predict": "/predict/",
        "about": "/about/",
        "how_to_use": "/how-to-use/",
        "knowledge_base": "/knowledge-base/",
        "disease_detail": "/disease/{}/",
        "disease_detail_kb": "/knowledge-base/disease/{}/",
    }


@pytest.fixture
def common_disease_names():
    """Фикстура распространенных названий заболеваний для тестирования страниц заболеваний."""
    return ["Грипп", "Простуда", "COVID-19", "Ангина"]


@pytest.fixture
def error_scenarios():
    """Фикстура сценариев которые могут вызвать ошибки для тестирования обработки исключений."""
    return {
        "invalid_symptoms": ["НесуществующийСимптом123", "AnotherFakeSymptom"],
        "special_chars": ["Симп%том@", "Тест#1", "Сим&птом"],
        "empty_data": [],
        "very_long_name": ["ОченьДлинноеНазваниеСимптома" * 10],
        "sql_injection": ["'; DROP TABLE diseases; --", "OR 1=1"],
        "xss_attempt": [
            '<script>alert("xss")</script>',
            "<img src=x onerror=alert(1)>",
        ],
    }


@pytest.fixture
def nonexistent_diseases():
    """Фикстура несуществующих названий заболеваний для тестирования обработки ошибок."""
    return [
        "НесуществующаяБолезнь123",
        "FakeDiseaseXYZ",
        "ТестоваяБолезньНеНайдена",
        "NonExistentDiseaseName",
    ]


@pytest.fixture
def invalid_urls():
    """Фикстура некорректных URL для тестирования обработки ошибок маршрутизации."""
    return [
        "/disease//",  # Пустое название болезни
        "/predict//",  # Некорректный URL
        "/knowledge-base/disease//",  # Пустое название в базе знаний
        "/invalid-page/",
        "/../../etc/passwd",  # Попытка обхода пути
        "/disease/<script>alert(1)</script>/",  # XSS в URL
    ]
