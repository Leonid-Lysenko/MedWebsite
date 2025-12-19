"""
Интеграционные тесты для основного потока диагностики приложения diagnosis.
Тестирует полные пользовательские сценарии и взаимодействие между компонентами системы.
Адаптировано для PostgreSQL.
"""

from unittest.mock import patch
import numpy as np
import pytest
from django.urls import reverse
from diagnosis.models import Disease, Symptom

# Эта фикстура автоматически даст доступ к БД всем тестам в этом файле
pytestmark = pytest.mark.django_db

@pytest.fixture
def setup_db_data():
    """Фикстура для подготовки данных в БД перед тестами."""
    # Создаем симптомы
    symptoms = ["Кашель", "Высокая температура", "Головная боль", "Насморк", "Чихание", "Усталость"]
    created_symptoms = []
    for s_name in symptoms:
        created_symptoms.append(Symptom.objects.create(name=s_name))
    
    # 1. Грипп (полный набор данных)
    Disease.objects.create(
        name="Грипп",
        description="Тестовое описание гриппа",
        treatment="Тестовое лечение гриппа",
        symptoms=["Высокая температура", "Кашель", "Головная боль"],
        severity="high",
        specialist="Терапевт",
        category="Инфекционные"
    )
    
    # 2. Простуда (тоже заполняем все обязательные поля)
    Disease.objects.create(
        name="Простуда",
        description="Описание простуды",
        treatment="Лечение простуды",
        symptoms=["Кашель", "Насморк"],
        severity="low",
        specialist="Терапевт",      
        category="Вирусные"         
    )
    
    # 3. COVID-19 (заглушки для обязательных полей)
    Disease.objects.create(
        name="COVID-19", 
        description="COVID desc",
        treatment="COVID treatment", 
        symptoms=["Кашель", "Температура"], 
        severity="high",             
        specialist="Инфекционист",   
        category="Инфекционные"      
    )
    
    # 4. СХУ
    Disease.objects.create(
        name="Синдром хронической усталости", 
        description="Fatigue desc",
        treatment="Rest",            
        symptoms=["Усталость"],      
        severity="medium",           
        specialist="Невролог",       
        category="Неврологические"   
    )
    
    # 5. ОРВИ
    Disease.objects.create(
        name="ОРВИ", 
        description="ORVI desc",
        treatment="Tea",             
        symptoms=["Насморк"],        
        severity="low",              
        specialist="Терапевт",       
        category="Вирусные"          
    )
    
    return created_symptoms


@pytest.mark.integration
def test_home_page_loads_correctly(client, setup_db_data):
    """Тест что главная страница загружается и содержит симптомы из БД."""
    response = client.get(reverse("home"))
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Медицинский Диагностический Помощник" in content
    # Проверяем, что симптомы из базы отобразились
    assert "Кашель" in content
    assert "Высокая температура" in content

@pytest.mark.integration
def test_symptom_selection_to_prediction(client, setup_db_data, common_symptoms):
    """Тест полного цикла: выбор симптомов -> отправка -> получение результатов."""
    with patch("diagnosis.views.model") as mock_model:
        # Вероятность 0.8 -> это > 0.1 -> должно быть "Высокая"
        mock_model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])
        mock_model.classes_ = np.array(["Грипп", "Простуда", "COVID-19"])
        
        with patch("diagnosis.views.diseases_list", ["Грипп", "Простуда", "COVID-19"]):
            response = client.post(reverse("predict"), {"symptoms": common_symptoms})
            
            assert response.status_code == 200
            content = response.content.decode("utf-8")
            
            # Проверяем наличие болезни
            assert "Грипп" in content
            # Проверяем текстовое описание вероятности вместо процентов
            assert "Высокая" in content
            
            # Проверяем описание из БД
            assert "Тестовое описание гриппа" in content

@pytest.mark.integration
def test_results_page_contains_disease_cards(client, setup_db_data, common_symptoms):
    """Тест что страница результатов содержит элементы интерфейса."""
    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[0.8, 0.15, 0.05]]
        mock_model.classes_ = np.array(["Грипп", "Простуда", "COVID-19"])
        
        with patch("diagnosis.views.diseases_list", ["Грипп", "Простуда", "COVID-19"]):
            response = client.post(reverse("predict"), {"symptoms": common_symptoms})
            
            assert response.status_code == 200
            content = response.content.decode("utf-8")
            assert "html" in content
            assert "body" in content
            assert "card" in content.lower() or "result" in content.lower()

@pytest.mark.integration
def test_navigation_from_results_to_disease_detail(client, setup_db_data):
    """Тест навигации от страницы результатов к деталям заболевания."""
    # Этот тест проверяет просто доступность страницы детализации, которая берет данные из БД
    disease_response = client.get(reverse("disease_detail", args=["Грипп"]))
    
    assert disease_response.status_code == 200
    disease_content = disease_response.content.decode("utf-8")
    
    assert "Грипп" in disease_content
    assert "Тестовое описание гриппа" in disease_content
    assert "не найдено" not in disease_content.lower()

@pytest.mark.integration
def test_empty_symptoms_submission(client):
    """Тест отправки формы без выбранных симптомов."""

    with patch("diagnosis.views.model") as mock_model:
        response = client.post(reverse("predict"), {"symptoms": []})
        
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        # Должна быть ошибка валидации
        assert "ошибк" in content.lower() or "выберите" in content.lower()

@pytest.mark.integration
def test_single_symptom_diagnosis(client, setup_db_data, minimal_symptoms):
    """Тест диагностики с одним симптомом."""
    with patch("diagnosis.views.model") as mock_model:
        # Вероятность 1.0 -> это > 0.1 -> должно быть "Высокая"
        mock_model.predict_proba.return_value = np.array([[1.0]])
        mock_model.classes_ = np.array(["Синдром хронической усталости"])
        
        with patch("diagnosis.views.diseases_list", ["Синдром хронической усталости"]):
            response = client.post(reverse("predict"), {"symptoms": minimal_symptoms})
            
            assert response.status_code == 200
            content = response.content.decode("utf-8")
            
            assert "Синдром хронической усталости" in content
            assert "Высокая" in content

@pytest.mark.integration
def test_multiple_symptoms_diagnosis(client, setup_db_data, respiratory_symptoms):
    """Тест диагностики с множеством симптомов."""
    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["ОРВИ"])
        
        with patch("diagnosis.views.diseases_list", ["ОРВИ"]):
            response = client.post(reverse("predict"), {"symptoms": respiratory_symptoms})
            
            assert response.status_code == 200
            content = response.content.decode("utf-8")
            assert "ОРВИ" in content

@pytest.mark.integration
def test_complete_user_journey(client, setup_db_data, common_symptoms):
    """Полный тест пользовательского сценария от начала до конца."""
    
    # 1. Главная страница (проверяем что симптомы грузятся из БД)
    home_response = client.get(reverse("home"))
    assert home_response.status_code == 200
    assert "Кашель" in home_response.content.decode("utf-8")

    # 2. Диагностика с симптомами
    with patch("diagnosis.views.model") as mock_model:
        mock_model.predict_proba.return_value = [[1.0]]
        mock_model.classes_ = np.array(["Грипп"])
        
        with patch("diagnosis.views.diseases_list", ["Грипп"]):
            predict_response = client.post(reverse("predict"), {"symptoms": common_symptoms})
            assert predict_response.status_code == 200
            assert "Грипп" in predict_response.content.decode("utf-8")

    # 3. Детали заболевания (из БД)
    detail_response = client.get(reverse("disease_detail", args=["Грипп"]))
    assert detail_response.status_code == 200
    assert "Тестовое лечение гриппа" in detail_response.content.decode("utf-8")

    # 4. Возврат к новой диагностике
    new_diagnosis_response = client.get(reverse("home"))
    assert new_diagnosis_response.status_code == 200
