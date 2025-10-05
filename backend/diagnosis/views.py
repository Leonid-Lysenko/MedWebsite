import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse

# Загружаем модель при старте приложения
try:
    model_data = joblib.load(settings.ML_MODEL_PATH)
    model = model_data['model']
    symptoms_list = model_data['features']
    diseases_list = model_data['classes']
    print("ML модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None
    symptoms_list = []
    diseases_list = []

def home(request):
    """Главная страница"""
    return render(request, 'diagnosis/home.html', {
        'symptoms_count': len(symptoms_list),
        'diseases_count': len(diseases_list)
    })

def predict(request):
    """Обработка предсказания"""
    if request.method == 'POST' and model is not None:
        try:
            # Получаем симптомы из формы
            selected_symptoms = request.POST.getlist('symptoms')
            
            # Создаем вектор для модели
            input_vector = np.zeros(len(symptoms_list))
            for symptom in selected_symptoms:
                if symptom in symptoms_list:
                    idx = symptoms_list.index(symptom)
                    input_vector[idx] = 1
            
            # Предсказание
            probabilities = model.predict_proba([input_vector])[0]
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            
            # Формируем результаты
            results = []
            for idx in top5_indices:
                results.append({
                    'disease': diseases_list[idx],
                    'probability': float(probabilities[idx]),
                    'percentage': f"{probabilities[idx]*100:.1f}%"
                })
            
            return render(request, 'diagnosis/results.html', {
                'results': results,
                'symptoms_count': len(selected_symptoms)
            })
            
        except Exception as e:
            return render(request, 'diagnosis/error.html', {
                'error': str(e)
            })
    
    return render(request, 'diagnosis/home.html')

def disease_detail(request, disease_name):
    """Детальная страница болезни"""
    # Заглушка
    return render(request, 'diagnosis/disease_detail.html', {
        'disease_name': disease_name
    })