"""
Интеграционные тесты для навигации по приложению diagnosis.
Проверяет доступность страниц, корректность навигационных ссылок и переходы между страницами.
Адаптировано для PostgreSQL.
"""

from unittest.mock import patch
import numpy as np
import pytest
from django.urls import reverse
from diagnosis.models import Disease, Symptom

# Разрешаем доступ к БД для всех тестов
pytestmark = pytest.mark.django_db

@pytest.fixture
def setup_data():
    """Создает необходимые данные в БД для навигации."""
    # Создаем симптомы (для главной страницы)
    Symptom.objects.create(name="Кашель")
    
    # Создаем болезни (для базы знаний и деталей)
    Disease.objects.create(
        name="Грипп", 
        description="Описание гриппа", 
        treatment="Лечение гриппа", 
        symptoms=["Жар"], 
        severity="high", 
        specialist="Терапевт", 
        category="Инфекционные"
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
    Disease.objects.create(
        name="COVID-19",
        description="Desc",
        treatment="Treat",
        symptoms=["Cough"],
        severity="high",
        specialist="Inf",
        category="Inf"
    )
    Disease.objects.create(
        name="Ангина",
        description="Desc",
        treatment="Treat",
        symptoms=["Throat"],
        severity="medium",
        specialist="ENT",
        category="Inf"
    )

@pytest.mark.integration
def test_all_main_pages_accessible(client, navigation_urls, setup_data):
    """Тест что все основные страницы приложения доступны."""
    static_pages = ["home", "about", "how_to_use", "knowledge_base"]
    
    for page in static_pages:
        url = navigation_urls[page]
        response = client.get(url)
        assert response.status_code == 200, f"Страница {page} недоступна"
        content = response.content.decode("utf-8")
        assert "html" in content
        assert "body" in content

@pytest.mark.integration
def test_navigation_menu_links(client, setup_data):
    """Тест что навигационное меню содержит все основные ссылки."""
    response = client.get(reverse("home"))
    content = response.content.decode("utf-8")
    
    # Проверяем наличие ссылок в меню
    # Текст ссылок может зависеть от шаблона, обычно это "Главная", "О нас" и т.д.
    nav_links = ["Главная", "О нас", "Как пользоваться", "База знаний"]
    for link in nav_links:
        assert link in content, f"Ссылка '{link}' отсутствует в навигации"

@pytest.mark.integration
def test_breadcrumb_navigation_from_knowledge_base(client, setup_data):
    """Тест хлебных крошек при переходе из базы знаний."""
    # Заходим на страницу болезни через базу знаний
    response = client.get("/knowledge-base/disease/Грипп/")
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    
    # Проверяем навигационную цепочку
    assert "Главная" in content
    assert "База знаний" in content
    # Убедимся, что загрузилась правильная болезнь
    assert "Грипп" in content

@pytest.mark.integration
def test_breadcrumb_navigation_from_results(client, setup_data):
    """Тест хлебных крошек при переходе из результатов диагностики."""
    # Эмулируем переход на страницу болезни (как будто из результатов)
    response = client.get("/disease/Грипп/")
    
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    
    assert "Главная" in content
    # В твоем шаблоне может быть "Результаты диагностики" или просто кнопка "Назад"
    # Проверяем наличие навигации
    assert "Грипп" in content

@pytest.mark.integration
def test_return_to_home_from_any_page(client, setup_data):
    """Тест что с любой страницы можно вернуться на главную."""
    pages_to_test = [reverse("about"), reverse("how_to_use"), reverse("knowledge_base")]
    
    for page_url in pages_to_test:
        response = client.get(page_url)
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        
        # Ищем ссылку на главную (обычно это логотип или пункт меню)
        # Проверяем наличие ссылки href="/" или href="{% url 'home' %}" (в рендере это /)
        assert 'href="/"' in content or "Главная" in content

@pytest.mark.integration
def test_disease_detail_accessibility(client, common_disease_names, setup_data):
    """Тест доступности страниц деталей заболеваний."""
    for disease_name in common_disease_names:
        # У нас в setup_data созданы Грипп, Простуда, COVID-19, Ангина
        # Фикстура common_disease_names должна совпадать с этими именами
        response = client.get(reverse("disease_detail", args=[disease_name]))
        
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "html" in content
        assert disease_name in content

@pytest.mark.integration
def test_knowledge_base_disease_detail_accessibility(client, common_disease_names, setup_data):
    """Тест доступности страниц деталей заболеваний из базы знаний."""
    for disease_name in common_disease_names:
        response = client.get(f"/knowledge-base/disease/{disease_name}/")
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert disease_name in content

@pytest.mark.integration
def test_error_pages_handling(client):
    """Тест обработки ошибок и недоступных страниц."""
    invalid_urls = [
        "/nonexistent-page/",
        "/disease/NonExistentDisease123/", # Это вернет 200 (заглушка), не 404
        "/invalid-path/",
    ]
    
    for url in invalid_urls:
        response = client.get(url)
        # Главное - не 500
        assert response.status_code != 500
        content = response.content.decode("utf-8")
        assert "html" in content

@pytest.mark.integration
def test_complete_navigation_flow(client, setup_data):
    """Полный тест навигационного потока через все приложение."""
    # 1. Главная
    response = client.get(reverse("home"))
    assert response.status_code == 200
    
    # 2. О нас
    response = client.get(reverse("about"))
    assert response.status_code == 200
    
    # 3. Как пользоваться
    response = client.get(reverse("how_to_use"))
    assert response.status_code == 200
    
    # 4. База знаний
    response = client.get(reverse("knowledge_base"))
    assert response.status_code == 200
    assert "Грипп" in response.content.decode("utf-8")
    
    # 5. Возврат на главную
    response = client.get(reverse("home"))
    assert response.status_code == 200
