import os

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from .disease_data import get_disease_info


def load_symptoms():
    """
    Загружает список симптомов из текстового файла.

    Returns:
        list: Список симптомов или пустой список в случае ошибки
    """
    try:
        symptoms_file = os.path.join(os.path.dirname(__file__), "translated_columns_list.txt")
        if os.path.exists(symptoms_file):
            with open(symptoms_file, "r", encoding="utf-8") as f:
                symptoms = [line.strip() for line in f if line.strip()]
            return symptoms
    except Exception as e:
        # Обработка ошибки чтения файла
        pass

    return []


# Глобальные переменные для отслеживания состояния системы
model = None
symptoms_list = []
diseases_list = []
model_loaded_successfully = False
model_error_message = ""


# Инициализация ML-модели и данных при запуске приложения
try:
    model = joblib.load(settings.ML_MODEL_PATH)
    symptoms_list = load_symptoms()
    diseases_list = model.classes_.tolist()
    model_loaded_successfully = True
    model_error_message = ""

except Exception as e:
    # В случае ошибки загрузки модели инициализируем пустые структуры
    model = None
    symptoms_list = load_symptoms()
    diseases_list = []
    model_loaded_successfully = False
    model_error_message = "система диагностики временно недоступна"


def home(request):
    """
    Главная страница приложения.

    Отображает форму для выбора симптомов и основную информацию о системе.
    Если модель не загружена, показывает предупреждение.
    """
    context = {
        "symptoms": symptoms_list,
        "symptoms_count": len(symptoms_list),
        "diseases_count": len(diseases_list) if diseases_list else 0,
        "model_loaded": model_loaded_successfully,
        "model_error": model_error_message,
    }

    return render(request, "diagnosis/home.html", context)


def predict(request):
    """
    Обработчик предсказания заболевания на основе выбранных симптомов.

    Args:
        request: HTTP-запрос с выбранными симптомами

    Returns:
        HttpResponse: Страница с результатами или ошибкой
    """

    if request.method == "POST" and model is not None:
        try:
            # Получаем список выбранных симптомов из формы
            selected_symptoms = request.POST.getlist("symptoms")

            # Создаем бинарный вектор симптомов для ML-модели
            input_vector = np.zeros(len(symptoms_list))

            # Заполняем вектор: 1 - симптом присутствует, 0 - отсутствует
            for symptom in selected_symptoms:
                if symptom in symptoms_list:
                    idx = symptoms_list.index(symptom)
                    input_vector[idx] = 1

            # Проверяем, что выбран хотя бы один симптом
            symptom_count = np.sum(input_vector)
            if symptom_count == 0:
                return render(
                    request,
                    "diagnosis/error.html",
                    {"error": "Пожалуйста, выберите хотя бы один симптом"},
                )

            # Получаем вероятности заболеваний от ML-модели
            probabilities = model.predict_proba([input_vector])[0]

            # Выбираем 5 заболеваний с наибольшей вероятностью
            top5_indices = np.argsort(probabilities)[-5:][::-1]

            # Формируем список результатов для отображения
            results = []
            for idx in top5_indices:
                disease_name = diseases_list[idx]
                probability = float(probabilities[idx])

                results.append(
                    {
                        "disease": disease_name,
                        "probability": probability,
                        "percentage": f"{probability * 100:.2f}%",
                        "description": get_disease_description(disease_name),
                        "treatment": get_disease_treatment(disease_name),
                    }
                )

            return render(
                request,
                "diagnosis/results.html",
                {
                    "results": results,
                    "symptoms_count": len(selected_symptoms),
                    "selected_symptoms": selected_symptoms,
                },
            )

        except Exception as e:
            # Обработка ошибок предсказания
            return render(
                request,
                "diagnosis/error.html",
                {"error": f"Произошла ошибка при анализе симптомов: {str(e)}"},
            )

    # Если запрос не POST или модель не загружена, возвращаем на главную
    return home(request)


def about(request):
    """Отображает страницу 'О проекте' с информацией о системе."""
    return render(request, "diagnosis/about.html")


def how_to_use(request):
    """Отображает страницу с инструкцией по использованию системы."""
    return render(request, "diagnosis/how_to_use.html")


def get_disease_description(disease_name):
    """
    Возвращает описание заболевания из базы знаний.

    Args:
        disease_name (str): Название заболевания

    Returns:
        str: Описание заболевания
    """
    disease_info = get_disease_info(disease_name)
    return disease_info["description"]


def get_disease_treatment(disease_name):
    """
    Возвращает рекомендации по лечению заболевания из базы знаний.

    Args:
        disease_name (str): Название заболевания

    Returns:
        str: Рекомендации по лечению
    """
    disease_info = get_disease_info(disease_name)
    return disease_info["treatment"]


def get_disease_suggestions(searched_name):
    """
    Возвращает список похожих названий заболеваний для подсказок при поиске.

    Args:
        searched_name (str): Искомое название заболевания

    Returns:
        list: Список похожих названий
    """
    from difflib import get_close_matches

    all_diseases = model.classes_.tolist() if model else []

    if not all_diseases:
        return []

    # Используем алгоритм нечеткого сравнения для поиска похожих названий
    suggestions = get_close_matches(searched_name, all_diseases, n=5, cutoff=0.3)
    return suggestions


def disease_detail(request, disease_name):
    """
    Отображает детальную страницу информации о заболевании.

    Args:
        request: HTTP-запрос
        disease_name (str): Название заболевания

    Returns:
        HttpResponse: Страница с детальной информацией или страница "не найдено"
    """
    disease_info = get_disease_info(disease_name)

    # Определяем источник перехода для корректного отображения навигации
    is_from_knowledge_base = "knowledge-base/disease" in request.path

    # Проверяем, есть ли информация о заболевании в базе знаний
    is_unknown_disease = disease_info["description"].startswith("Информация о заболевании")

    if is_unknown_disease:
        # Если заболевание не найдено, показываем страницу с подсказками
        return render(
            request,
            "diagnosis/disease_not_found.html",
            {
                "searched_disease": disease_name,
                "suggestions": get_disease_suggestions(disease_name) if model else [],
            },
        )

    # Отображаем детальную страницу с информацией о заболевании
    return render(
        request,
        "diagnosis/disease_detail.html",
        {
            "disease_name": disease_name,
            "description": disease_info["description"],
            "treatment": disease_info["treatment"],
            "symptoms": disease_info["symptoms"],
            "severity": disease_info["severity"],
            "specialist": disease_info["specialist"],
            "category": disease_info["category"],
            "emergency_contacts": [
                "112 - Единая служба спасения",
                "103 - Скорая помощь",
                "03 - Скорая помощь (старый номер)",
            ],
            "is_from_knowledge_base": is_from_knowledge_base,
        },
    )


def knowledge_base(request):
    """
    Отображает страницу базы знаний со всеми заболеваниями.

    Группирует заболевания по алфавиту для удобной навигации.
    """
    # Получаем все заболевания из ML-модели
    all_diseases = model.classes_.tolist() if model else []

    # Загружаем дополнительную информацию для каждого заболевания
    diseases_with_info = []
    for disease_name in sorted(all_diseases):
        disease_info = get_disease_info(disease_name)

        # Преобразуем код серьезности в читаемый текст
        severity_text = get_severity_display(disease_info["severity"])

        diseases_with_info.append(
            {
                "name": disease_name,
                "info": disease_info,
                "severity_display": severity_text,
            }
        )

    # Группируем заболевания по первой букве для алфавитного указателя
    diseases_by_letter = {}
    for disease in diseases_with_info:
        first_letter = disease["name"][0].upper() if disease["name"] else ""
        if first_letter not in diseases_by_letter:
            diseases_by_letter[first_letter] = []
        diseases_by_letter[first_letter].append(disease)

    return render(
        request,
        "diagnosis/knowledge_base.html",
        {"diseases_by_letter": diseases_by_letter, "total_diseases": len(all_diseases)},
    )


def get_severity_display(severity):
    """
    Преобразует код серьезности заболевания в читаемое представление.

    Args:
        severity (str): Код серьезности ('high', 'medium', 'low', etc.)

    Returns:
        str: Текстовое представление серьезности
    """
    severity_map = {
        "high": "ВЫСОКАЯ",
        "medium": "СРЕДНЯЯ",
        "low": "НИЗКАЯ",
        "unknown": "НЕОПРЕДЕЛЕНА",
        "variable": "ЗАВИСИТ ОТ СТАДИИ",
    }
    return severity_map.get(severity, "НЕОПРЕДЕЛЕНА")
