"""
Unit-тесты для модуля views.py приложения diagnosis.
Тестирует функции представлений, вспомогательные функции и обработку запросов.
Адаптировано для работы с PostgreSQL (ORM).
"""

import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest
from django.http import Http404

# Импортируем сам модуль views, чтобы иметь доступ к глобальным переменным при необходимости
import diagnosis.views
from diagnosis.views import (
    about,
    disease_detail,
    get_disease_description,
    get_disease_suggestions,
    get_disease_treatment,
    get_severity_display,
    home,
    how_to_use,
    knowledge_base,
    get_ml_symptoms,  # Заменили load_symptoms на новую функцию
    predict,
)

# ===== Тесты функции get_ml_symptoms (бывшая load_symptoms) =====

@pytest.mark.unit
@patch("diagnosis.views.Symptom.objects")
def test_get_ml_symptoms_returns_list(mock_symptom_manager):
    """Тест проверяет, что get_ml_symptoms возвращает список строк из БД."""
    # Настраиваем цепочку вызовов ORM: Symptom.objects.all().order_by("id").values_list("name", flat=True)
    mock_queryset = Mock()
    mock_queryset.order_by.return_value.values_list.return_value = ["headache", "fever"]
    mock_symptom_manager.all.return_value = mock_queryset

    result = get_ml_symptoms()
    
    assert isinstance(result, list)
    assert result == ["headache", "fever"]
    # Проверяем, что методы вызывались правильно
    mock_symptom_manager.all.assert_called_once()
    mock_queryset.order_by.assert_called_with("id")

# ===== Тесты функции get_severity_display =====

@pytest.mark.unit
def test_severity_display_all_cases(severity_test_cases):
    """Тест всех вариантов преобразования кодов серьезности в текст."""
    for severity, expected in severity_test_cases:
        assert get_severity_display(severity) == expected

# ===== Тесты вспомогательных функций информации о заболеваниях =====

@pytest.mark.unit
@patch("diagnosis.views.get_disease_info_from_db")
def test_get_disease_description(mock_get_info):
    """Тест получения описания заболевания."""
    mock_get_info.return_value = {"description": "Тестовое описание"}
    
    description = get_disease_description("Грипп")
    
    assert description == "Тестовое описание"
    mock_get_info.assert_called_once_with("Грипп")

@pytest.mark.unit
@patch("diagnosis.views.get_disease_info_from_db")
def test_get_disease_treatment(mock_get_info):
    """Тест получения лечения заболевания."""
    mock_get_info.return_value = {"treatment": "Тестовое лечение"}
    
    treatment = get_disease_treatment("Грипп")
    
    assert treatment == "Тестовое лечение"
    mock_get_info.assert_called_once_with("Грипп")

# ===== Тесты функции подсказок заболеваний =====

@pytest.mark.unit
def test_get_disease_suggestions_with_model(mock_model_classes):
    """Тест подсказок заболеваний с работающей моделью."""
    # Патчим глобальную переменную model в модуле views
    mock_model_instance = Mock()
    mock_model_instance.classes_ = mock_model_classes
    
    with patch("diagnosis.views.model", mock_model_instance):
        suggestions = get_disease_suggestions("Грип")
        assert isinstance(suggestions, list)
        # difflib должен найти "Грипп" для "Грип"
        assert "Грипп" in suggestions

@pytest.mark.unit
def test_get_disease_suggestions_no_model():
    """Тест подсказок заболеваний без модели."""
    with patch("diagnosis.views.model", None):
        suggestions = get_disease_suggestions("Грипп")
        assert suggestions == []

# ===== Тесты home view =====

@pytest.mark.unit
@patch("diagnosis.views.get_ml_symptoms") # Мокаем получение симптомов
def test_home_view_returns_200(mock_get_symptoms, client):
    """Тест проверяет, что home view возвращает статус 200."""
    mock_get_symptoms.return_value = ["симптомы"]
    response = client.get("/")
    assert response.status_code == 200

@pytest.mark.unit
@patch("diagnosis.views.render")
@patch("diagnosis.views.get_ml_symptoms")
@patch("diagnosis.views.model_loaded_successfully", True)
@patch("diagnosis.views.diseases_list", ["Грипп", "Простуда"])
def test_home_view_context_structure(
    mock_get_symptoms, mock_render, mock_symptoms_list_fixture, factory
):
    """
    Тест home view проверяет правильность формирования контекста.
    """
    # Arrange
    mock_get_symptoms.return_value = mock_symptoms_list_fixture
    mock_render.return_value = Mock() # HttpResponse
    request = factory.get("/")

    # Act
    home(request)

    # Assert
    # Получаем контекст, который был передан в render
    call_args = mock_render.call_args
    context_arg = call_args[0][2] # третий аргумент - контекст

    assert "symptoms" in context_arg
    assert "symptoms_by_letter" in context_arg
    assert context_arg["symptoms_count"] == len(mock_symptoms_list_fixture)
    
    # Проверка группировки
    symptoms_by_letter = context_arg["symptoms_by_letter"]
    assert isinstance(symptoms_by_letter, dict)
    # Проверяем, что ключи отсортированы (в Python 3.7+ словари сохраняют порядок вставки, 
    # а в views.py мы явно сортируем ключи перед вставкой)
    keys = list(symptoms_by_letter.keys())
    assert keys == sorted(keys)

@pytest.mark.unit
@patch("diagnosis.views.render")
@patch("diagnosis.views.get_ml_symptoms")
def test_home_view_symptoms_grouping_logic(
    mock_get_symptoms, mock_render, mock_symptoms_with_special_chars, factory
):
    """Тест логики группировки симптомов (спецсимволы и цифры в '#')."""
    # Arrange
    mock_get_symptoms.return_value = mock_symptoms_with_special_chars
    mock_render.return_value = Mock()
    request = factory.get("/")

    # Act
    with patch("diagnosis.views.model_loaded_successfully", True):
        home(request)

    # Assert
    context_arg = mock_render.call_args[0][2]
    
    # Проверяем наличие группы для специальных символов
    assert "#" in context_arg["symptoms_by_letter"]
    
    hash_group = context_arg["symptoms_by_letter"].get("#", [])
    # Проверяем, что симптомы, начинающиеся не с кириллицы, попали сюда
    # В твоей фикстуре это "1 стадия", "@симптом" и т.д.
    assert any("1" in s for s in hash_group)

# ===== Тесты knowledge_base view =====

@pytest.mark.unit
@patch("diagnosis.views.Disease.objects")
def test_knowledge_base_returns_200(mock_disease_manager, client):
    """Тест проверяет, что knowledge_base возвращает статус 200 и делает запрос к БД."""
    # Настраиваем мок для Disease.objects.order_by("name").all()
    mock_queryset = MagicMock()
    # Имитируем итерацию по queryset (пустой список)
    mock_queryset.__iter__.return_value = iter([]) 
    mock_queryset.__len__.return_value = 0
    
    mock_disease_manager.order_by.return_value.all.return_value = mock_queryset
    
    response = client.get("/knowledge-base/")
    assert response.status_code == 200

@pytest.mark.unit
@patch("diagnosis.views.Disease.objects")
def test_knowledge_base_context_data(mock_disease_manager, client):
    """Тест проверяет, что данные из БД попадают в контекст."""
    # Создаем мок объекта заболевания
    mock_disease = Mock()
    mock_disease.name = "Ангина"
    mock_disease.description = "Описание"
    mock_disease.treatment = "Лечение"
    mock_disease.symptoms = ["Боль"]
    mock_disease.severity = "medium"
    mock_disease.specialist = "ЛОР"
    mock_disease.category = "Инфекция"

    # Настраиваем QuerySet
    mock_queryset = MagicMock()
    mock_queryset.__iter__.return_value = iter([mock_disease])
    mock_queryset.__len__.return_value = 1
    
    mock_disease_manager.order_by.return_value.all.return_value = mock_queryset

    response = client.get("/knowledge-base/")
    
    assert response.status_code == 200
    # Проверяем, что заболевание попало в группу 'А'
    context = response.context
    assert "А" in context["diseases_by_letter"]
    disease_data = context["diseases_by_letter"]["А"][0]
    assert disease_data["name"] == "Ангина"

# ===== Тесты disease_detail view =====

@pytest.mark.unit
@patch("diagnosis.views.Disease.objects")
def test_disease_detail_found(mock_disease_manager, client):
    """Тест отображения существующего заболевания."""
    mock_disease = Mock()
    mock_disease.name = "Грипп"
    mock_disease.description = "Описание гриппа"
    mock_disease.treatment = "Лечение гриппа"
    mock_disease.symptoms = ["Жар"]
    mock_disease.severity = "high"
    mock_disease.specialist = "Терапевт"
    mock_disease.category = "ОРВИ"

    # get должен вернуть этот объект
    mock_disease_manager.get.return_value = mock_disease

    response = client.get("/disease/Грипп/")
    
    assert response.status_code == 200
    assert "Грипп" in response.content.decode("utf-8")
    mock_disease_manager.get.assert_called_with(name__iexact="Грипп")

@pytest.mark.unit
@patch("diagnosis.views.Disease.objects")
def test_disease_detail_not_found_in_db(mock_disease_manager, client):
    """Тест обработки случая, когда заболевания нет в БД."""
    # Имитируем исключение DoesNotExist
    # Важно: нужно правильно сэмулировать исключение ORM
    mock_disease_manager.get.side_effect = diagnosis.views.Disease.DoesNotExist
    
    # При отсутствии болезни view должна вернуть страницу "не найдено" (которая технически 200 OK с другим шаблоном)
    # но с заглушкой "Информация ... готовится"
    
    response = client.get("/disease/Несуществующая/")
    assert response.status_code == 200
    
    content = response.content.decode("utf-8")
    # Проверяем, что отобразился шаблон disease_not_found.html или disease_detail с заглушкой
    # В твоем коде при DoesNotExist возвращается словарь с "Информация... готовится", 
    # и затем проверяется if is_unknown_disease
    assert "Информация о заболевании" in content or "не найдено" in content.lower()

# ===== Тесты predict view =====

@pytest.mark.unit
def test_predict_get_returns_home(client):
    """Тест GET запроса к predict возвращает home страницу (фактически редирект или вызов home)."""
    # В твоем коде return home(request), значит вернется 200 OK (страница home)
    with patch("diagnosis.views.get_ml_symptoms", return_value=[]):
         response = client.get("/predict/")
         assert response.status_code == 200

@pytest.mark.unit
@patch("diagnosis.views.model")
@patch("diagnosis.views.get_ml_symptoms")
@patch("diagnosis.views.get_disease_info_from_db")
def test_predict_success_flow(
    mock_get_db_info, mock_get_symptoms, mock_model, factory, predict_request_data
):
    """Тест полного цикла успешного предсказания."""
    # 1. Мокаем симптомы из БД
    symptoms_db = ["кашель", "температура", "боль"]
    mock_get_symptoms.return_value = symptoms_db
    
    # 2. Мокаем модель
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
    mock_model.classes_ = np.array(["Грипп", "Простуда"])
    
    # 3. Мокаем получение информации о болезни (вызывается внутри цикла results)
    mock_get_db_info.return_value = {
        "description": "Desc", "treatment": "Treat", "severity": "low"
    }

    # 4. Мокаем глобальную переменную diseases_list (она используется для маппинга индексов)
    with patch("diagnosis.views.diseases_list", ["Грипп", "Простуда"]):
        request = factory.post("/predict/", predict_request_data) # {"symptoms": ["кашель", "температура"]}
        response = predict(request)
        
        assert response.status_code == 200
        # Проверяем, что вектор был создан верно
        # У нас 3 симптома в БД, 2 выбрано. Вектор должен быть [1, 1, 0] (если порядок совпал)
        # predict_proba вызывается с вектором
        args, _ = mock_model.predict_proba.call_args
        input_vector = args[0][0]
        assert len(input_vector) == 3
        assert input_vector[0] == 1 # кашель есть
        assert input_vector[1] == 1 # температура есть
        assert input_vector[2] == 0 # боли нет

@pytest.mark.unit
@patch("diagnosis.views.model")
@patch("diagnosis.views.get_ml_symptoms")
def test_predict_no_symptoms_selected(mock_get_symptoms, mock_model, factory):
    """Тест: пользователь отправил форму без симптомов."""
    mock_get_symptoms.return_value = ["с1", "с2"]
    
    # Пустой список симптомов
    request = factory.post("/predict/", {"symptoms": []})
    
    response = predict(request)
    assert response.status_code == 200
    # Должен отрендерить error.html
    # Так как мы не мокаем render полностью, проверяем контент (если это integration тест) 
    # или просто статус, подразумевая, что не упало.
    # Но лучше проверить наличие сообщения об ошибке, если используем Client.
    # В случае factory render возвращает HttpResponse с контентом
    content = response.content.decode("utf-8")
    assert "ошибк" in content.lower() or "выберите" in content.lower()

# ===== Тесты глобальных переменных (smoke) =====

@pytest.mark.unit
def test_module_globals_exist():
    """Проверка наличия критических глобальных переменных."""
    assert hasattr(diagnosis.views, "model")
    assert hasattr(diagnosis.views, "diseases_list")
    assert hasattr(diagnosis.views, "model_loaded_successfully")
    # symptoms_list больше не существует глобально, тест на него удаляем

# ===== Тесты простых view (Smoke tests) =====

@pytest.mark.unit
def test_about_view(client):
    """Тест about view возвращает корректный статус."""
    response = client.get("/about/")
    assert response.status_code == 200

@pytest.mark.unit
def test_how_to_use_view(client):
    """Тест how_to_use view возвращает корректный статус."""
    response = client.get("/how-to-use/")
    assert response.status_code == 200

# ===== Тесты производительности (Адаптировано под БД) =====

@pytest.mark.unit
def test_predict_performance(client):
    """
    Тест производительности предсказания ML-модели.
    Проверяет, что предсказание работает быстрее 10 секунд (включая работу БД).
    """
    with (
        patch("diagnosis.views.model") as mock_model,
        patch("diagnosis.views.get_ml_symptoms"),
        patch("diagnosis.views.diseases_list", ["Грипп", "Простуда"]),
        patch("diagnosis.views.get_disease_info_from_db") as mock_db_info
    ):
        # Настраиваем моки
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_model.classes_ = ["Грипп", "Простуда"]
        mock_db_info.return_value = {
            "description": "Desc", "treatment": "Treat", "severity": "low"
        }
        
        # Измеряем время
        start_time = time.time()
        response = client.post("/predict/", {"symptoms": ["кашель", "температура"]})
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert response.status_code == 200
        assert execution_time < 10.0, f"Время выполнения слишком велико: {execution_time:.2f}с"
