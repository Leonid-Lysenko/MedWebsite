import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
import os
from .disease_data import get_disease_info

# Загружаем реальные симптомы
def load_symptoms():
    """Загружает переведенные симптомы из файла"""
    try:
        symptoms_file = os.path.join(os.path.dirname(__file__), 'translated_columns_list.txt')
        if os.path.exists(symptoms_file):
            with open(symptoms_file, 'r', encoding='utf-8') as f:
                symptoms = [line.strip() for line in f if line.strip()]
            print(f"Загружено {len(symptoms)} симптомов")
            return symptoms
    except Exception as e:
        print(f"Ошибка загрузки симптомов: {e}")
    
    return []

# Загружаем модель и данные при старте приложения
try:
    model = joblib.load(settings.ML_MODEL_PATH)
    print("Модель RandomForest загружена успешно!")
    
    # Загружаем симптомы
    symptoms_list = load_symptoms()
    
    # Берем заболевания напрямую из модели (они уже на русском!)
    diseases_list = model.classes_.tolist()
    
    print(f"Заболеваний в модели: {len(diseases_list)}")
    print(f"Симптомов: {len(symptoms_list)}")
    
    # Покажем несколько заболеваний для проверки
    print("Первые 10 заболеваний из модели:")
    for i, disease in enumerate(diseases_list[:10]):
        print(f"   {i}. {disease}")
    
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None
    symptoms_list = load_symptoms()
    diseases_list = []

def home(request):
    """Главная страница с формой ввода симптомов"""
    return render(request, 'diagnosis/home.html', {
        'symptoms': symptoms_list,
        'symptoms_count': len(symptoms_list),
        'diseases_count': len(diseases_list) if diseases_list else 0
    })

def predict(request):
    """Обработка предсказания заболевания с использованием реальной ML модели"""
    if request.method == 'POST' and model is not None:
        try:
            # Получаем симптомы из формы
            selected_symptoms = request.POST.getlist('symptoms')
            age = request.POST.get('age', '')
            gender = request.POST.get('gender', '')
            
            print(f"Получены симптомы: {selected_symptoms}")
            print(f"Возраст: {age}, Пол: {gender}")
            
            # Создаем вектор для модели
            input_vector = np.zeros(len(symptoms_list))
            
            for symptom in selected_symptoms:
                if symptom in symptoms_list:
                    idx = symptoms_list.index(symptom)
                    input_vector[idx] = 1
                    print(f"Симптом '{symptom}' добавлен в позиции {idx}")
            
            symptom_count = np.sum(input_vector)
            print(f"Всего активных симптомов в векторе: {symptom_count}")
            
            if symptom_count == 0:
                return render(request, 'diagnosis/error.html', {
                    'error': 'Пожалуйста, выберите хотя бы один симптом'
                })
            
            print("Запускаем реальное предсказание модели...")
            
            # Получаем вероятности от модели
            probabilities = model.predict_proba([input_vector])[0]
            
            # Берем топ-5 заболеваний
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            
            print("Топ-5 вероятных заболеваний:")
            
            # Формируем результаты - используем названия напрямую из модели
            results = []
            for idx in top5_indices:
                disease_name = diseases_list[idx]
                probability = float(probabilities[idx])
                
                print(f"   {idx}. {disease_name}: {probability:.4f} ({probability*100:.2f}%)")
                
                results.append({
                    'disease': disease_name,
                    'probability': probability,
                    'percentage': f"{probability*100:.2f}%",
                    'description': get_disease_description(disease_name),
                    'treatment': get_disease_treatment(disease_name),
                })
            
            print(f"Всего ненулевых вероятностей: {np.sum(probabilities > 0)}")
            print(f"Максимальная вероятность: {np.max(probabilities):.4f}")
            
            return render(request, 'diagnosis/results.html', {
                'results': results,
                'symptoms_count': len(selected_symptoms),
                'age': age,
                'gender': gender
            })
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            import traceback
            print(f"Детали ошибки: {traceback.format_exc()}")
            
            return render(request, 'diagnosis/error.html', {
                'error': f'Произошла ошибка при анализе симптомов: {str(e)}'
            })
    
    return home(request)

def about(request):
    """Страница 'О нас'"""
    return render(request, 'diagnosis/about.html')

# временная описательная информация (скоро придумаем что-то получше !)

from .disease_data import get_disease_info

# Обновил функции:
def get_disease_description(disease_name):
    """Возвращает описание болезни из базы знаний"""
    disease_info = get_disease_info(disease_name)
    return disease_info["description"]

def get_disease_treatment(disease_name):
    """Возвращает рекомендации по лечению из базы знаний"""
    disease_info = get_disease_info(disease_name)
    return disease_info["treatment"]

# Обновил функцию disease_detail:
def disease_detail(request, disease_name):
    """Детальная страница болезни"""
    disease_info = get_disease_info(disease_name)
    
    return render(request, 'diagnosis/disease_detail.html', {
        'disease_name': disease_name,
        'description': disease_info["description"],
        'treatment': disease_info["treatment"],
        'symptoms': disease_info["symptoms"],
        'severity': disease_info["severity"],
        'specialist': disease_info["specialist"],
        'category': disease_info["category"],
        'emergency_contacts': ['112 - Единая служба спасения', '103 - Скорая помощь', '03 - Скорая помощь (старый номер)']
    })