"""
Фикстуры для unit-тестов приложения diagnosis.

Содержит набор фикстур для тестирования views, disease_data и других модулей.
"""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from django.test import Client, RequestFactory


@pytest.fixture
def client():
    """Фикстура Django test client для тестирования HTTP-запросов."""
    return Client()


@pytest.fixture
def factory():
    """Фикстура RequestFactory для создания тестовых запросов."""
    return RequestFactory()


@pytest.fixture
def mock_disease_info():
    """Фикстура мока информации о заболевании для тестирования."""
    return {
        "description": "Тестовое описание болезни",
        "treatment": "Тестовое лечение болезни",
        "symptoms": ["симптом1", "симптом2"],
        "severity": "medium",
        "specialist": "Терапевт",
        "category": "Инфекционные",
    }


@pytest.fixture
def mock_file_content():
    """Фикстура содержимого файла симптомов для тестирования загрузки данных."""
    return ["headache\n", "fever\n", "cough\n", "fatigue\n"]


@pytest.fixture
def predict_request_data():
    """Фикстура данных для POST запроса predict view."""
    return {"symptoms": ["кашель", "температура"]}


@pytest.fixture
def severity_test_cases():
    """Фикстура тестовых случаев для преобразования кодов серьезности в текст."""
    return [
        ("high", "ВЫСОКАЯ"),
        ("medium", "СРЕДНЯЯ"),
        ("low", "НИЗКАЯ"),
        ("unknown", "НЕОПРЕДЕЛЕНА"),
        ("variable", "ЗАВИСИТ ОТ СТАДИИ"),
        ("invalid", "НЕОПРЕДЕЛЕНА"),
    ]


@pytest.fixture
def symptom_combinations():
    """Фикстура различных комбинаций симптомов для тестирования предсказаний."""
    return [
        ["кашель"],  # один симптом
        ["кашель", "температура"],  # два симптома
        ["кашель", "температура", "головная боль"],  # три симптома
    ]


@pytest.fixture
def disease_paths():
    """Фикстура путей для тестирования disease_detail view из разных источников."""
    return [
        "/disease/Грипп/",  # из результатов диагностики
        "/knowledge-base/disease/Грипп/",  # из базы знаний
    ]


@pytest.fixture
def mock_model_classes():
    """Фикстура для мока classes_ атрибута ML модели."""
    mock_array = Mock()
    mock_array.tolist.return_value = [
        "Грипп",
        "Грипп A",
        "Грипп B",
        "Простуда",
        "COVID-19",
    ]
    return mock_array


@pytest.fixture
def probability_test_cases():
    """Фикстура тестовых случаев вероятностей заболеваний от ML модели."""
    return np.array([0.854321, 0.123456, 0.022223])


@pytest.fixture
def mock_file_operations():
    """Фикстура для моков файловых операций при тестировании загрузки симптомов."""

    class FileMocks:
        def __init__(self):
            self.mock_exists = Mock()
            self.mock_open = Mock()
            self.mock_file = MagicMock()

    mocks = FileMocks()
    mocks.mock_file.__enter__.return_value = mocks.mock_file
    mocks.mock_file.__exit__.return_value = None
    return mocks


# Фикстуры для test_disease_data_units.py


@pytest.fixture
def edge_case_disease_names():
    """Фикстура с пограничными случаями названий заболеваний для тестирования обработки."""
    return [
        "",  # пустая строка
        "   ",  # пробелы
        "НесуществующееЗаболевание123",  # несуществующее заболевание
        "ГРИПП",  # верхний регистр
        "грипп",  # нижний регистр
        " Грипп ",  # с пробелами
        "Covid-19",  # другая раскладка
    ]
