"""
Фикстуры для E2E тестов приложения diagnosis.

Содержит фикстуры для настройки браузера, сервера и утилит тестирования.
"""

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait


@pytest.fixture
def browser():
    """Фикстура для запуска браузера Chrome в режиме тестирования."""
    chrome_options = Options()

    # Настройки для headless режима
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    # Отключаем логирование для чистоты вывода
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)

        print("Браузер Chrome успешно запущен")
        yield driver

    except Exception as e:
        pytest.skip(f"Не удалось запустить Chrome: {e}")
    finally:
        if "driver" in locals():
            driver.quit()
            print("Браузер закрыт")


@pytest.fixture
def live_server_url():
    """Фикстура URL запущенного Django сервера для тестирования."""
    return "http://localhost:8000"


@pytest.fixture
def wait():
    """Фикстура для создания объектов явных ожиданий в тестах."""

    def _wait(driver, timeout=10):
        return WebDriverWait(driver, timeout)

    return _wait
