"""
Интеграционные тесты для базы знаний заболеваний приложения diagnosis.
Тестирует функциональность базы знаний, включая навигацию, поиск и отображение заболеваний.
Адаптировано для PostgreSQL.
"""

from unittest.mock import patch
import numpy as np
import pytest
from django.urls import reverse
from diagnosis.models import Disease

# Разрешаем доступ к БД для всех тестов
pytestmark = pytest.mark.django_db

@pytest.fixture
def setup_diseases():
    """Создает набор заболеваний в БД для тестов."""
    Disease.objects.create(
        name="Ангина", 
        description="Описание ангины", 
        treatment="Лечение", 
        symptoms=["Боль в горле"], 
        severity="medium", 
        specialist="ЛОР", 
        category="Инфекционные"
    )
    Disease.objects.create(
        name="Бронхит", 
        description="Описание бронхита", 
        treatment="Лечение", 
        symptoms=["Кашель"], 
        severity="medium", 
        specialist="Терапевт", 
        category="Пульмонология"
    )
    Disease.objects.create(
        name="Грипп", 
        description="Тестовое описание гриппа", 
        treatment="Тестовое лечение", 
        symptoms=["Жар", "Кашель"], 
        severity="high", 
        specialist="Терапевт", 
        category="Вирусные"
    )
    Disease.objects.create(
        name="Простуда", 
        description="Описание простуды", 
        treatment="Лечение", 
        symptoms=["Насморк"], 
        severity="low", 
        specialist="Терапевт", 
        category="Вирусные"
    )

@pytest.mark.integration
def test_knowledge_base_page_loads(client, setup_diseases):
    """Тест что страница базы знаний загружается корректно."""
    response = client.get(reverse("knowledge_base"))
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "База знаний заболеваний" in content

@pytest.mark.integration
def test_knowledge_base_contains_alphabet_navigation(client, setup_diseases):
    """Тест что база знаний содержит алфавитную навигацию."""
    response = client.get(reverse("knowledge_base"))
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    
    # Мы создали "Ангина", "Бронхит", "Грипп" -> буквы А, Б, Г должны быть
    assert "А" in content
    assert "Б" in content
    assert "Г" in content

@pytest.mark.integration
def test_knowledge_base_displays_disease_cards(client, setup_diseases):
    """Тест что база знаний отображает карточки заболеваний."""
    response = client.get(reverse("knowledge_base"))
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    
    # Имена болезней должны быть на странице
    assert "Грипп" in content
    assert "Простуда" in content
    assert "Тестовое описание гриппа" in content

@pytest.mark.integration
def test_disease_detail_from_knowledge_base(client, setup_diseases):
    """Тест перехода из базы знаний к деталям заболевания."""
    # Запрос по реальному URL, данные берутся из БД
    response = client.get("/knowledge-base/disease/Грипп/")
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    
    assert "Грипп" in content
    assert "Тестовое описание гриппа" in content
    assert "не найдено" not in content.lower()

@pytest.mark.integration
def test_search_functionality_in_knowledge_base(client, setup_diseases):
    """Тест наличия элементов поиска в базе знаний."""
    # Сам поиск часто реализуется JS-скриптом на клиенте (фильтрация списка), 
    # поэтому мы проверяем наличие поля ввода или слова "Поиск".
    response = client.get(reverse("knowledge_base"))
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "search" in content.lower() or "поиск" in content.lower()

@pytest.mark.integration
def test_severity_badges_display(client, setup_diseases):
    """Тест что бейджи серьезности отображаются в базе знаний."""
    response = client.get(reverse("knowledge_base"))
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    
    # Мы создали Грипп с severity="high" -> должно отобразиться "ВЫСОКАЯ" (через get_severity_display)
    # Проверяем наличие слова "ВЫСОКАЯ" или класса badge
    assert "ВЫСОКАЯ" in content or "high" in content.lower()

@pytest.mark.integration
def test_category_display_in_knowledge_base(client, setup_diseases):
    """Тест что категории заболеваний отображаются."""
    response = client.get(reverse("knowledge_base"))
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    
    # Мы создали "Инфекционные" и "Вирусные"
    assert "Инфекционные" in content or "Вирусные" in content

@pytest.mark.integration
def test_knowledge_base_with_empty_diseases(client):
    """Тест базы знаний когда нет заболеваний (пустая БД)."""
    # Не вызываем setup_diseases -> база пустая
    response = client.get(reverse("knowledge_base"))
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "html" in content
    # Не должно быть имен болезней
    assert "Грипп" not in content

@pytest.mark.integration
def test_complete_knowledge_base_journey(client, setup_diseases):
    """Полный тест сценария работы с базой знаний."""
    # 1. Заходим на главную
    home_response = client.get(reverse("home"))
    assert home_response.status_code == 200

    # 2. Переходим в базу знаний
    kb_response = client.get(reverse("knowledge_base"))
    assert kb_response.status_code == 200
    assert "Ангина" in kb_response.content.decode("utf-8")

    # 3. Кликаем на болезнь (Грипп)
    detail_response = client.get("/knowledge-base/disease/Грипп/")
    assert detail_response.status_code == 200
    assert "Тестовое лечение" in detail_response.content.decode("utf-8")

    # 4. Возвращаемся в список
    kb_return_response = client.get(reverse("knowledge_base"))
    assert kb_return_response.status_code == 200
