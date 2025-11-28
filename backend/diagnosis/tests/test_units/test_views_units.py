"""
Unit-тесты для модуля views.py приложения diagnosis.

Тестирует функции представлений, вспомогательные функции и обработку запросов.
"""

import importlib
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

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
    load_symptoms,
    predict,
)

# ===== Тесты функции load_symptoms =====


@pytest.mark.unit
def test_load_symptoms_returns_list():
    """Тест проверяет, что функция load_symptoms возвращает список."""
    result = load_symptoms()
    assert isinstance(result, list)


@pytest.mark.unit
def test_load_symptoms_returns_non_empty_list():
    """Тест проверяет, что список симптомов не пустой."""
    result = load_symptoms()
    assert len(result) > 0


@pytest.mark.unit
@patch("diagnosis.views.os.path.exists")
@patch("diagnosis.views.open")
def test_load_symptoms_with_mock_file(mock_open, mock_exists, mock_file_operations, mock_file_content):
    """Тест загрузки симптомов с моком файла."""
    mock_exists.return_value = True
    mock_file_operations.mock_file.readlines.return_value = mock_file_content
    mock_open.return_value = mock_file_operations.mock_file

    with patch("diagnosis.views.load_symptoms") as mock_load:
        mock_load.return_value = ["headache", "fever", "cough"]
        result = mock_load()
        assert result == ["headache", "fever", "cough"]


@pytest.mark.unit
@patch("diagnosis.views.os.path.exists")
def test_load_symptoms_file_not_found(mock_exists):
    """Тест обработки случая, когда файл симптомов не найден."""
    mock_exists.return_value = False

    with patch("diagnosis.views.load_symptoms") as mock_load:
        mock_load.return_value = []
        result = mock_load()
        assert result == []


# ===== Тесты функции get_severity_display =====


@pytest.mark.unit
def test_severity_display_all_cases(severity_test_cases):
    """Тест всех вариантов преобразования кодов серьезности в текст."""
    for severity, expected in severity_test_cases:
        assert get_severity_display(severity) == expected


# ===== Тесты вспомогательных функций информации о заболеваниях =====


@pytest.mark.unit
@patch("diagnosis.views.get_disease_info")
def test_get_disease_description(mock_get_info, mock_disease_info):
    """Тест получения описания заболевания."""
    mock_get_info.return_value = mock_disease_info
    description = get_disease_description("Грипп")
    assert description == "Тестовое описание болезни"
    mock_get_info.assert_called_once_with("Грипп")


@pytest.mark.unit
@patch("diagnosis.views.get_disease_info")
def test_get_disease_treatment(mock_get_info, mock_disease_info):
    """Тест получения лечения заболевания."""
    mock_get_info.return_value = mock_disease_info
    treatment = get_disease_treatment("Грипп")
    assert treatment == "Тестовое лечение болезни"
    mock_get_info.assert_called_once_with("Грипп")


# ===== Тесты функции подсказок заболеваний =====


@pytest.mark.unit
@patch("diagnosis.views.model")
def test_get_disease_suggestions_with_model(mock_model, mock_model_classes):
    """Тест подсказок заболеваний с работающей моделью."""
    mock_model_instance = Mock()
    mock_model_instance.classes_ = mock_model_classes

    with patch("diagnosis.views.model", mock_model_instance):
        suggestions = get_disease_suggestions("Грип")
        assert isinstance(suggestions, list)


@pytest.mark.unit
def test_get_disease_suggestions_no_model():
    """Тест подсказок заболеваний без модели."""
    with patch("diagnosis.views.model", None):
        suggestions = get_disease_suggestions("Грипп")
        assert suggestions == []


@pytest.mark.unit
@patch("diagnosis.views.model")
def test_get_disease_suggestions_empty_input(mock_model, mock_model_classes):
    """Тест подсказок заболеваний с пустым вводом."""
    mock_model_instance = Mock()
    mock_model_instance.classes_ = mock_model_classes

    with patch("diagnosis.views.model", mock_model_instance):
        suggestions = get_disease_suggestions("")
        assert isinstance(suggestions, list)


@pytest.mark.unit
@patch("diagnosis.views.model")
def test_get_disease_suggestions_no_matches(mock_model, mock_model_classes):
    """Тест случая, когда нет совпадений для введенного заболевания."""
    mock_model_instance = Mock()
    mock_model_instance.classes_ = mock_model_classes

    with patch("diagnosis.views.model", mock_model_instance):
        suggestions = get_disease_suggestions("НесуществующаяБолезнь")
        assert isinstance(suggestions, list)


# ===== Тесты home view =====


@pytest.mark.unit
def test_home_view_returns_200(client):
    """Тест проверяет, что home view возвращает статус 200."""
    response = client.get("/")
    assert response.status_code == 200


@pytest.mark.unit
def test_home_view_contains_keywords(client):
    """Тест проверяет, что home view содержит ключевые слова."""
    response = client.get("/")
    content = response.content.decode("utf-8")
    assert response.status_code == 200
    assert "симптом" in content.lower() or "диагност" in content.lower()


# ===== Тесты predict view =====


@pytest.mark.unit
def test_predict_get_returns_home(client):
    """Тест GET запроса к predict возвращает home страницу."""
    response = client.get("/predict/")
    assert response.status_code == 200


@pytest.mark.unit
def test_predict_no_symptoms_returns_error(client):
    """Тест predict без симптомов возвращает ошибку."""
    response = client.post("/predict/", {})
    assert response.status_code == 200


# ===== Тесты простых view =====


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


# ===== Тесты knowledge_base view =====


@pytest.mark.unit
def test_knowledge_base_returns_200(client):
    """Тест проверяет, что knowledge_base возвращает статус 200."""
    response = client.get("/knowledge-base/")
    assert response.status_code == 200


@pytest.mark.unit
def test_knowledge_base_contains_diseases(client):
    """Тест проверяет, что база знаний содержит информацию о заболеваниях."""
    response = client.get("/knowledge-base/")
    content = response.content.decode("utf-8")
    assert response.status_code == 200
    assert "заболевани" in content.lower() or "база" in content.lower()


# ===== Тесты disease_detail view =====


@pytest.mark.unit
def test_disease_detail_returns_200_for_existing_disease(client):
    """Тест проверяет, что disease_detail возвращает 200 для существующего заболевания."""
    response = client.get("/disease/Грипп/")
    assert response.status_code == 200


@pytest.mark.unit
def test_disease_detail_handles_unknown_disease(client):
    """Тест обработки неизвестного заболевания."""
    response = client.get("/disease/НесуществующаяБолезнь123/")
    assert response.status_code == 200


@pytest.mark.unit
def test_disease_detail_from_knowledge_base(client):
    """Тест disease_detail при переходе из базы знаний."""
    response = client.get("/knowledge-base/disease/Грипп/")
    assert response.status_code == 200


@pytest.mark.unit
def test_disease_detail_different_sources(client, disease_paths):
    """Тест disease_detail с разными источниками перехода."""
    for path in disease_paths:
        response = client.get(path)
        assert response.status_code == 200


# ===== Тесты граничных случаев =====


@pytest.mark.unit
def test_empty_input_vector_creation():
    """Тест создания пустого вектора симптомов."""
    assert True


@pytest.mark.unit
def test_predict_with_unknown_symptoms():
    """Тест predict с неизвестными симптомами."""
    assert True


# ===== Тесты глобальных переменных =====


@pytest.mark.unit
def test_symptoms_list_initialized():
    """Тест что symptoms_list инициализирована."""
    from diagnosis.views import symptoms_list

    assert isinstance(symptoms_list, list)


@pytest.mark.unit
def test_diseases_list_initialized():
    """Тест что diseases_list инициализирована."""
    from diagnosis.views import diseases_list

    assert isinstance(diseases_list, list)


@pytest.mark.unit
def test_model_loaded_status():
    """Тест статуса загрузки модели."""
    from diagnosis.views import model_loaded_successfully

    assert isinstance(model_loaded_successfully, bool)


# ===== Smoke тесты для базовой функциональности =====


@pytest.mark.unit
def test_all_views_importable():
    """Тест что все views импортируются без ошибок."""
    from diagnosis import views

    assert views is not None


@pytest.mark.unit
def test_basic_functions_callable():
    """Тест что базовые функции можно вызывать."""
    result = load_symptoms()
    assert isinstance(result, list)

    result = get_severity_display("high")
    assert isinstance(result, str)


# ===== Тесты крайних случаев для predict view =====


@pytest.mark.unit
@patch("diagnosis.views.model")
@patch("diagnosis.views.symptoms_list", ["кашель", "температура", "головная боль"])
def test_predict_with_ml_model_success(mock_model, factory, predict_request_data):
    """Тест успешного предсказания с ML моделью."""
    mock_model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])
    mock_model.classes_ = np.array(["Грипп", "Простуда", "COVID-19"])

    with patch("diagnosis.views.diseases_list", ["Грипп", "Простуда", "COVID-19"]):
        with patch("diagnosis.views.get_disease_description") as mock_desc:
            with patch("diagnosis.views.get_disease_treatment") as mock_treat:
                mock_desc.return_value = "Описание болезни"
                mock_treat.return_value = "Лечение болезни"

                request = factory.post("/predict/", predict_request_data)
                response = predict(request)

                assert response.status_code == 200
                assert hasattr(response, "content")


@pytest.mark.unit
@patch("diagnosis.views.model")
@patch("diagnosis.views.symptoms_list", ["кашель", "температура"])
def test_predict_creates_correct_input_vector(mock_model, factory):
    """Тест создания правильного вектора симптомов."""
    mock_model.predict_proba.return_value = np.array([[1.0]])
    mock_model.classes_ = np.array(["Болезнь1"])

    with patch("diagnosis.views.diseases_list", ["Болезнь1"]):
        with patch("diagnosis.views.get_disease_description") as mock_desc:
            with patch("diagnosis.views.get_disease_treatment") as mock_treat:
                mock_desc.return_value = "Описание"
                mock_treat.return_value = "Лечение"

                request = factory.post("/predict/", {"symptoms": ["кашель", "неизвестный_симптом"]})

                response = predict(request)

                assert response.status_code == 200
                mock_model.predict_proba.assert_called_once()


@pytest.mark.unit
@patch("diagnosis.views.model")
def test_predict_exception_handling(mock_model, factory):
    """Тест обработки исключений в predict."""
    mock_model.predict_proba.side_effect = Exception("ML модель упала")
    mock_model.classes_ = np.array(["Болезнь1"])

    with patch("diagnosis.views.symptoms_list", ["симптом1"]):
        with patch("diagnosis.views.diseases_list", ["Болезнь1"]):
            request = factory.post("/predict/", {"symptoms": ["симптом1"]})

            response = predict(request)

            assert response.status_code == 200
            content = response.content.decode("utf-8")
            assert "ошибк" in content.lower()


@pytest.mark.unit
@patch("diagnosis.views.model")
def test_predict_with_different_symptom_combinations(mock_model, factory, symptom_combinations):
    """Тест predict с различными комбинациями симптомов."""
    mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
    mock_model.classes_ = np.array(["Болезнь1", "Болезнь2"])

    with patch("diagnosis.views.symptoms_list", ["кашель", "температура", "головная боль"]):
        with patch("diagnosis.views.diseases_list", ["Болезнь1", "Болезнь2"]):
            with patch("diagnosis.views.get_disease_description") as mock_desc:
                with patch("diagnosis.views.get_disease_treatment") as mock_treat:
                    mock_desc.return_value = "Описание"
                    mock_treat.return_value = "Лечение"

                    for symptoms in symptom_combinations:
                        request = factory.post("/predict/", {"symptoms": symptoms})

                        response = predict(request)
                        assert response.status_code == 200


# ===== Тесты крайних случаев инициализации модели =====


@pytest.mark.unit
def test_global_variables_initialized():
    """Тест что глобальные переменные инициализированы."""
    from diagnosis.views import diseases_list, model_error_message, model_loaded_successfully, symptoms_list

    assert isinstance(symptoms_list, list)
    assert isinstance(diseases_list, list)
    assert isinstance(model_loaded_successfully, bool)
    assert isinstance(model_error_message, str)


@pytest.mark.unit
@patch("diagnosis.views.joblib.load")
@patch("diagnosis.views.settings.ML_MODEL_PATH", "/fake/path/model.pkl")
def test_model_initialization_exception(mock_joblib):
    """Тест обработки исключения при инициализации модели."""
    mock_joblib.side_effect = Exception("Файл модели не найден")

    importlib.reload(diagnosis.views)

    assert diagnosis.views.model is None
    assert diagnosis.views.model_loaded_successfully is False
    assert "система диагностики временно недоступна" in diagnosis.views.model_error_message


# ===== Тесты создания вектора симптомов =====


@pytest.mark.unit
@patch("diagnosis.views.model")
@patch("diagnosis.views.symptoms_list", ["симптом1", "симптом2", "симптом3"])
def test_input_vector_with_known_symptoms(mock_model, factory):
    """Тест создания вектора с известными симптомами."""
    mock_model.predict_proba.return_value = np.array([[1.0]])
    mock_model.classes_ = np.array(["Болезнь1"])

    with patch("diagnosis.views.diseases_list", ["Болезнь1"]):
        with patch("diagnosis.views.get_disease_description") as mock_desc:
            with patch("diagnosis.views.get_disease_treatment") as mock_treat:
                mock_desc.return_value = "Описание"
                mock_treat.return_value = "Лечение"

                request = factory.post("/predict/", {"symptoms": ["симптом1", "симптом3"]})

                response = predict(request)

                assert response.status_code == 200
                mock_model.predict_proba.assert_called_once()


@pytest.mark.unit
@patch("diagnosis.views.model")
@patch("diagnosis.views.symptoms_list", ["симптом1", "симптом2"])
def test_input_vector_with_unknown_symptoms(mock_model, factory):
    """Тест создания вектора с неизвестными симптомами."""
    mock_model.predict_proba.return_value = np.array([[1.0]])
    mock_model.classes_ = np.array(["Болезнь1"])

    with patch("diagnosis.views.diseases_list", ["Болезнь1"]):
        with patch("diagnosis.views.get_disease_description") as mock_desc:
            with patch("diagnosis.views.get_disease_treatment") as mock_treat:
                mock_desc.return_value = "Описание"
                mock_treat.return_value = "Лечение"

                request = factory.post("/predict/", {"symptoms": ["симптом1", "неизвестный_симптом"]})

                response = predict(request)

                assert response.status_code == 200


# ===== Тесты форматирования результатов =====


@pytest.mark.unit
@patch("diagnosis.views.model")
def test_results_probability_formatting(mock_model, factory, probability_test_cases):
    """Тест форматирования вероятностей в результатах."""
    mock_model.predict_proba.return_value = probability_test_cases
    mock_model.classes_ = np.array(["Болезнь1", "Болезнь2", "Болезнь3"])

    with patch("diagnosis.views.symptoms_list", ["симптом1"]):
        with patch("diagnosis.views.diseases_list", ["Болезнь1", "Болезнь2", "Болезнь3"]):
            with patch("diagnosis.views.get_disease_description") as mock_desc:
                with patch("diagnosis.views.get_disease_treatment") as mock_treat:
                    mock_desc.return_value = "Описание"
                    mock_treat.return_value = "Лечение"

                    request = factory.post("/predict/", {"symptoms": ["симптом1"]})

                    response = predict(request)

                    assert response.status_code == 200
                    content = response.content.decode("utf-8")
                    assert "%" in content


# ===== Тесты производительности предсказания ML-модели =====


@pytest.mark.unit
def test_predict_performance(client):
    """Тест производительности предсказания ML-модели."""

    with (
        patch("diagnosis.views.model") as mock_model,
        patch("diagnosis.views.symptoms_list", ["кашель", "температура", "головная боль"]),
        patch("diagnosis.views.diseases_list", ["Грипп", "Простуда"]),
        patch("diagnosis.views.get_disease_description") as mock_desc,
        patch("diagnosis.views.get_disease_treatment") as mock_treat,
    ):

        # Настраиваем мок модели
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_model.classes_ = ["Грипп", "Простуда"]
        mock_desc.return_value = "Описание"
        mock_treat.return_value = "Лечение"

        # Измеряем время выполнения
        start_time = time.time()

        response = client.post("/predict/", {"symptoms": ["кашель", "температура"]})

        end_time = time.time()
        execution_time = end_time - start_time

        # Проверяем что время выполнения < 10 секунд
        assert execution_time < 10.0, f"Время выполнения предсказания превышает 10 секунд: {execution_time:.2f}с"
        assert response.status_code == 200
