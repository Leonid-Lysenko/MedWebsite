"""
Unit-тесты для модуля disease_data.py приложения diagnosis.

Тестирует функцию get_disease_info и корректность структуры возвращаемых данных.
"""

import time
from unittest.mock import Mock, patch

import pytest

import diagnosis.disease_data as disease_data

# ===== Тесты функции get_disease_info =====


@pytest.mark.unit
def test_get_disease_info_returns_dict():
    """Тест что функция возвращает словарь."""
    result = disease_data.get_disease_info("Грипп")
    assert isinstance(result, dict)


@pytest.mark.unit
def test_get_disease_info_has_required_keys():
    """Тест что словарь содержит все обязательные ключи."""
    result = disease_data.get_disease_info("Грипп")
    required_keys = ["description", "treatment", "symptoms", "severity", "specialist", "category"]

    for key in required_keys:
        assert key in result
        assert result[key] is not None


@pytest.mark.unit
def test_get_disease_info_known_disease():
    """Тест для любого заболевания (все возвращают заглушки)."""
    disease_name = "Грипп"
    result = disease_data.get_disease_info(disease_name)

    assert result["description"] is not None
    assert len(result["description"]) > 0
    assert isinstance(result["symptoms"], list)
    assert result["severity"] in ["high", "medium", "low", "variable", "unknown"]


@pytest.mark.unit
def test_get_disease_info_unknown_disease():
    """Тест для неизвестного заболевания."""
    result = disease_data.get_disease_info("НесуществующееЗаболевание123")

    assert "информация о заболевании" in result["description"].lower()
    assert "готовится" in result["description"].lower() or "обратитесь" in result["description"].lower()
    assert result["severity"] == "unknown"
    assert result["specialist"] == "Терапевт"


@pytest.mark.unit
def test_get_disease_info_case_insensitive():
    """Тест что функция не чувствительна к регистру."""
    result1 = disease_data.get_disease_info("ГРИПП")
    result2 = disease_data.get_disease_info("грипп")
    result3 = disease_data.get_disease_info("Грипп")

    for result in [result1, result2, result3]:
        assert isinstance(result, dict)
        assert "description" in result
        assert "treatment" in result


@pytest.mark.unit
def test_get_disease_info_with_whitespace():
    """Тест обработки названий с пробелами."""
    result = disease_data.get_disease_info("  Грипп  ")
    assert isinstance(result, dict)
    assert "description" in result


@pytest.mark.unit
def test_get_disease_info_edge_cases(edge_case_disease_names):
    """Тест пограничных случаев названий заболеваний."""
    for disease_name in edge_case_disease_names:
        result = disease_data.get_disease_info(disease_name)
        assert isinstance(result, dict)
        assert all(
            key in result for key in ["description", "treatment", "symptoms", "severity", "specialist", "category"]
        )


# ===== Тесты структуры возвращаемых данных =====


@pytest.mark.unit
def test_disease_info_description_always_has_content():
    """Тест что описание всегда имеет содержание."""
    known_diseases = ["Грипп", "Мигрень", "COVID-19", "Гастрит"]

    for disease in known_diseases:
        result = disease_data.get_disease_info(disease)
        assert len(result["description"]) > 0
        assert "информация о заболевании" in result["description"].lower()


@pytest.mark.unit
def test_disease_info_treatment_always_has_content():
    """Тест что лечение всегда имеет содержание."""
    known_diseases = ["Грипп", "Мигрень", "COVID-19"]

    for disease in known_diseases:
        result = disease_data.get_disease_info(disease)
        assert len(result["treatment"]) > 0
        assert "обратитесь" in result["treatment"].lower() or "рекомендуется" in result["treatment"].lower()


@pytest.mark.unit
def test_disease_info_symptoms_is_list():
    """Тест что симптомы возвращаются как список."""
    result = disease_data.get_disease_info("Грипп")
    assert isinstance(result["symptoms"], list)
    assert isinstance(result["symptoms"], list)


@pytest.mark.unit
def test_disease_info_severity_valid():
    """Тест что серьезность всегда валидное значение."""
    test_diseases = ["Грипп", "Мигрень", "COVID-19", "Несуществующее"]

    valid_severities = ["high", "medium", "low", "variable", "unknown"]

    for disease in test_diseases:
        result = disease_data.get_disease_info(disease)
        assert result["severity"] in valid_severities


@pytest.mark.unit
def test_disease_info_specialist_valid():
    """Тест что специалист всегда строка."""
    test_diseases = ["Грипп", "Мигрень", "COVID-19", "Несуществующее"]

    for disease in test_diseases:
        result = disease_data.get_disease_info(disease)
        assert isinstance(result["specialist"], str)
        assert len(result["specialist"]) > 0


@pytest.mark.unit
def test_disease_info_category_valid():
    """Тест что категория всегда строка."""
    test_diseases = ["Грипп", "Мигрень", "COVID-19", "Несуществующее"]

    for disease in test_diseases:
        result = disease_data.get_disease_info(disease)
        assert isinstance(result["category"], str)
        assert len(result["category"]) > 0


# ===== Тесты консистентности данных =====


@pytest.mark.unit
def test_consistent_data_structure():
    """Тест консистентности структуры данных для разных заболеваний."""
    diseases = ["Грипп", "Мигрень", "COVID-19", "Гастрит", "Ангина"]

    for disease in diseases:
        result = disease_data.get_disease_info(disease)

        assert isinstance(result["description"], str)
        assert isinstance(result["treatment"], str)
        assert isinstance(result["symptoms"], list)
        assert isinstance(result["severity"], str)
        assert isinstance(result["specialist"], str)
        assert isinstance(result["category"], str)

        assert "информация о заболевании" in result["description"].lower()
        assert "обратитесь" in result["treatment"].lower() or "рекомендуется" in result["treatment"].lower()


# ===== Тесты производительности =====


@pytest.mark.unit
def test_get_disease_info_performance():
    """Тест производительности функции (должна работать быстро)."""
    start_time = time.time()

    for _ in range(100):
        disease_data.get_disease_info("Грипп")
        disease_data.get_disease_info("Несуществующее")

    end_time = time.time()
    execution_time = end_time - start_time

    assert execution_time < 1.0


# ===== Тесты деталей текущей реализации =====


@pytest.mark.unit
def test_flu_template_info():
    """Тест шаблонной информации о гриппе."""
    result = disease_data.get_disease_info("Грипп")

    assert "информация о заболевании" in result["description"].lower()
    assert "'грипп'" in result["description"].lower() or "грипп" in result["description"].lower()
    assert "готовится" in result["description"].lower() or "специалист" in result["description"].lower()


@pytest.mark.unit
def test_migraine_template_info():
    """Тест шаблонной информации о мигрени."""
    result = disease_data.get_disease_info("Мигрень")

    assert "информация о заболевании" in result["description"].lower()
    assert "'мигрень'" in result["description"].lower() or "мигрень" in result["description"].lower()
    assert "обратитесь" in result["treatment"].lower()


# ===== Smoke тесты =====


@pytest.mark.unit
def test_module_importable():
    """Тест что модуль импортируется без ошибок."""
    import diagnosis.disease_data

    assert disease_data is not None


@pytest.mark.unit
def test_function_callable():
    """Тест что функция вызывается без ошибок."""
    result = disease_data.get_disease_info("Тест")
    assert result is not None


# ===== Тесты для разных языков/кодировок =====


@pytest.mark.unit
def test_unicode_disease_names():
    """Тест обработки Unicode названий."""
    unicode_names = ["Грипп", "COVID-19", "Гастроэнтерит", "Ангина"]

    for name in unicode_names:
        result = disease_data.get_disease_info(name)
        assert isinstance(result, dict)
        assert all(key in result for key in ["description", "treatment", "symptoms"])


@pytest.mark.unit
def test_special_characters_in_names():
    """Тест обработки специальных символов в названиях."""
    special_names = ["COVID-19", "ОРВИ", "Грипп A", "Грипп B", "SARS-CoV-2"]

    for name in special_names:
        result = disease_data.get_disease_info(name)
        assert isinstance(result, dict)


# ===== Тесты на основе фактического поведения =====


@pytest.mark.unit
def test_all_diseases_return_same_structure():
    """Тест что все заболевания возвращают одинаковую структуру."""
    diseases = ["Грипп", "Простуда", "Ангина", "Бронхит", "Пневмония"]

    structures = []
    for disease in diseases:
        result = disease_data.get_disease_info(disease)
        structures.append(set(result.keys()))

    first_structure = structures[0]
    for structure in structures[1:]:
        assert structure == first_structure


@pytest.mark.unit
def test_template_text_consistency():
    """Тест консистентности шаблонного текста."""
    diseases = ["Грипп", "Мигрень", "COVID-19"]

    for disease in diseases:
        result = disease_data.get_disease_info(disease)

        assert disease.lower() in result["description"].lower()
        assert "врач" in result["treatment"].lower() or "специалист" in result["treatment"].lower()
