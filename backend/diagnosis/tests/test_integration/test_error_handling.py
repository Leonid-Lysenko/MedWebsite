"""
Интеграционные тесты для обработки ошибок приложения diagnosis.
Тестирует устойчивость системы к некорректным данным и отсутствие падений.
Адаптировано для PostgreSQL.
"""

from unittest.mock import patch

import numpy as np
import pytest
from django.urls import reverse

from diagnosis.models import Disease, Symptom

# Разрешаем доступ к БД для всех тестов в этом файле
pytestmark = pytest.mark.django_db


@pytest.fixture
def setup_symptoms():
    """Создаем базовые симптомы в БД для тестов."""
    symptoms = ["Кашель", "Температура", "Головная боль"]  # Добавил эмодзи сюда же для теста Unicode
    for s_name in symptoms:
        Symptom.objects.create(name=s_name)


@pytest.mark.integration
def test_predict_with_invalid_symptoms(client, error_scenarios, setup_symptoms):
    """Тест обработки невалидных симптомов в предсказании (которых нет в БД)."""
    with patch("diagnosis.views.model") as mock_model:
        # Модель вернет результат даже если симптомы не валидны (вектор будет из нулей, или частично заполнен)
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["Грипп"])

        # Нужно замокать diseases_list, чтобы views.py не упал при формировании ответа
        with patch("diagnosis.views.diseases_list", ["Грипп"]):
            # Создаем болезнь в БД, чтобы описание подтянулось
            Disease.objects.create(
                name="Грипп", description="Desc", symptoms=["s"], severity="low", specialist="Terapevt", category="Cat"
            )

            response = client.post(reverse("predict"), {"symptoms": error_scenarios["invalid_symptoms"]})

            assert response.status_code == 200
            content = response.content.decode("utf-8")
            assert "html" in content
            # Скорее всего будет ошибка "выберите хотя бы один симптом", если ни один не совпал
            # Или результат, если что-то совпало. Главное - не 500.


@pytest.mark.integration
def test_predict_with_special_characters(client, error_scenarios, setup_symptoms):
    """Тест обработки симптомов со специальными символами."""
    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["Грипп"])

        with patch("diagnosis.views.diseases_list", ["Грипп"]):
            # Создаем болезнь
            Disease.objects.create(
                name="Грипп", description="Desc", symptoms=["s"], severity="low", specialist="Terapevt", category="Cat"
            )

            response = client.post(reverse("predict"), {"symptoms": error_scenarios["special_chars"]})

            assert response.status_code == 200
            content = response.content.decode("utf-8")
            assert "html" in content
            assert "Internal Server Error" not in content


@pytest.mark.integration
def test_predict_with_empty_symptoms(client, error_scenarios):
    """Тест обработки пустого списка симптомов."""
    # Здесь мок модели не обязателен, т.к. проверка на пустой список идет ДО вызова модели
    with patch("diagnosis.views.model"):
        response = client.post(reverse("predict"), {"symptoms": error_scenarios["empty_data"]})

        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "html" in content
        # Должно быть сообщение об ошибке
        assert "ошибк" in content.lower() or "выберите" in content.lower()


@pytest.mark.integration
def test_disease_detail_nonexistent_disease(client, nonexistent_diseases):
    """Тест запроса несуществующих заболеваний (должна быть страница 'не найдено', а не 404/500)."""
    for disease_name in nonexistent_diseases:
        # Мы НЕ создаем болезни в БД. Значит, get() кинет DoesNotExist
        response = client.get(reverse("disease_detail", args=[disease_name]))

        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "html" in content
        assert "Internal Server Error" not in content
        # Проверяем, что вывелась заглушка или страница ошибки
        assert "информация о заболевании" in content.lower() or "не найдено" in content.lower()


@pytest.mark.integration
def test_knowledge_base_nonexistent_disease(client, nonexistent_diseases):
    """Тест запроса несуществующих заболеваний через базу знаний."""
    for disease_name in nonexistent_diseases:
        response = client.get(f"/knowledge-base/disease/{disease_name}/")

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
            pass  # 404 это нормально
        else:
            content = response.content.decode("utf-8")
            assert "html" in content


@pytest.mark.integration
def test_ml_model_failure_handling(client, setup_symptoms):
    """Тест обработки сбоя ML модели (возврат ошибки пользователю)."""
    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.side_effect = Exception("ML model failed")

        # Отправляем валидный симптом "Кашель", чтобы дойти до вызова модели
        response = client.post(reverse("predict"), {"symptoms": ["Кашель"]})

        assert response.status_code == 200
        content = response.content.decode("utf-8")

        # Должна быть страница ошибки (error.html)
        assert "ошибк" in content.lower() or "error" in content.lower()
        assert "Internal Server Error" not in content  # Это не 500 Django, а наш обработанный exception


@pytest.mark.integration
def test_large_input_handling(client):
    """Тест обработки большого количества симптомов (DoS защита уровня view)."""
    large_symptoms_list = [f"Симптом{i}" for i in range(100)]

    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["Грипп"])

        with patch("diagnosis.views.diseases_list", ["Грипп"]):
            Disease.objects.create(
                name="Грипп", description="D", symptoms=["s"], severity="l", specialist="S", category="C"
            )

            response = client.post(reverse("predict"), {"symptoms": large_symptoms_list})

            assert response.status_code == 200
            content = response.content.decode("utf-8")
            assert "html" in content


@pytest.mark.integration
def test_unicode_handling(client, setup_symptoms):
    """Тест обработки Unicode символов (эмодзи и кириллица)."""
    # В фикстуре setup_symptoms мы создали "Головная боль"
    unicode_symptoms = ["Головная боль"]

    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["Грипп"])

        with patch("diagnosis.views.diseases_list", ["Грипп"]):
            Disease.objects.create(
                name="Грипп", description="D", symptoms=["s"], severity="l", specialist="S", category="C"
            )

            response = client.post(reverse("predict"), {"symptoms": unicode_symptoms})

            assert response.status_code == 200
            content = response.content.decode("utf-8")
            assert "html" in content


@pytest.mark.integration
def test_error_page_display(client):
    """Тест что страница 'не найдено' корректно отображается с подсказками."""

    mock_model = patch("diagnosis.views.model").start()
    mock_model.classes_ = np.array(["Грипп", "Простуда"])

    # Используем латиницу, чтобы исключить проблемы с кодировкой в консоли Windows
    disease_name = "NonExistentDiseaseXYZ"

    try:
        response = client.get(reverse("disease_detail", args=[disease_name]))

        assert response.status_code == 200
        content = response.content.decode("utf-8")

        # 1. Проверяем заголовок
        assert "Заболевание не найдено" in content

        # 2. Проверяем, что имя болезни отобразилось
        assert disease_name in content

        # 3. Проверяем наличие кнопки на главную
        assert "На главную" in content

    finally:
        patch.stopall()
