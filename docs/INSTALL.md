# Руководство по установке

## Минимальные требования
- Python 3.12+
- pip (входит в состав Python)

## Установка за 3 шага

### 1. Клонирование и настройка окружения
```bash
git clone <repository-url>
cd MedWebsite
python -m venv venv
```

**Активация окружения:**

```bash
#Windows:
venv\Scripts\activate

# Linux/MacOS:
source venv/bin/activate
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Запуск системы
```bash
cd backend
python manage.py runserver
```
**Система будет доступна по адресу:** http://localhost:8000

## Проверка работоспособности

**После запуска:**

   - Откройте главную страницу в браузере
   - Выберите несколько симптомов
   - Нажмите "Анализировать симптомы"
   - Убедитесь в корректном отображении результатов

## Запуск тестов

```bash
cd backend
pytest
```