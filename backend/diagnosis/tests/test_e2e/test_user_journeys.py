"""
E2E тесты полных пользовательских сценариев приложения diagnosis.
Тестирует реальные пользовательские сценарии через браузер Chrome.
"""

import time
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

@pytest.mark.e2e
def test_main_diagnosis_flow(browser, live_server_url, wait):
    """E2E тест основного потока диагностики: главная → выбор симптомов → результаты."""
    try:
        browser.get(live_server_url)

        wait(browser).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        assert "Медицинский" in browser.page_source

        browser.execute_script("window.scrollTo(0, 300);")
        time.sleep(1)

        # 1. Поиск и выбор симптомов
        symptom_checkboxes = browser.find_elements(By.CSS_SELECTOR, 'input[type="checkbox"][name="symptoms"]')
        
        # ВАЖНО: Если симптомов нет, тест должен упасть, а не пройти молча
        assert len(symptom_checkboxes) > 0, "На странице не найдены симптомы! Проверьте, что БД заполнена."

        selected_count = 0
        for i, checkbox in enumerate(symptom_checkboxes):
            try:
                browser.execute_script("arguments[0].scrollIntoView(true);", checkbox)
                time.sleep(0.1)
                
                if checkbox.is_displayed() and checkbox.is_enabled():
                    browser.execute_script("arguments[0].click();", checkbox)
                    selected_count += 1
                    if selected_count >= 3:
                        break
            except Exception:
                continue

        assert selected_count > 0, "Не удалось выбрать ни одного симптома"

        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        # 2. Нажатие кнопки отправки
        button_selectors = ['button[type="submit"]', ".btn-medical", "button"]
        target_button = None
        for selector in button_selectors:
            buttons = browser.find_elements(By.CSS_SELECTOR, selector)
            for button in buttons:
                if button.is_displayed() and button.is_enabled():
                    target_button = button
                    break
            if target_button:
                break
        
        assert target_button is not None, "Кнопка отправки формы не найдена"
        
        browser.execute_script("arguments[0].scrollIntoView(true);", target_button)
        time.sleep(0.5)
        browser.execute_script("arguments[0].click();", target_button)

        # 3. Ожидание результатов
        time.sleep(3)
        
        page_content_lower = browser.page_source.lower()
        
        # Проверяем, что мы не на главной (URL изменился или контент)
        # Ищем признаки страницы результатов
        is_result_page = "результат" in page_content_lower or "вероятн" in page_content_lower
        assert is_result_page, "Не перешли на страницу результатов"

    except Exception as e:
        try:
            browser.save_screenshot("e2e_test1_error.png")
        except:
            pass
        pytest.fail(f"E2E тест провален: {e}")

@pytest.mark.e2e
def test_knowledge_base_navigation(browser, live_server_url, wait):
    """E2E тест навигации по базе знаний: главная → база знаний → детали заболевания."""
    try:
        # 1. Переход в базу знаний
        browser.get(f"{live_server_url}/knowledge-base/")

        wait(browser).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        page_content = browser.page_source
        assert len(page_content) > 0, "Страница базы знаний пустая"

        # 2. Поиск карточек заболеваний
        card_selectors = [".disease-card", ".card", '[class*="disease"]']
        disease_cards = []
        for selector in card_selectors:
            cards = browser.find_elements(By.CSS_SELECTOR, selector)
            if cards:
                disease_cards.extend(cards)
                break
        
        # ВАЖНО: Тест должен упасть, если база пустая
        assert len(disease_cards) > 0, "В базе знаний нет карточек заболеваний!"

        # 3. Клик по ссылке на заболевание
        link_selectors = ['a[href*="/disease/"]', 'a[href*="knowledge-base/disease/"]']
        disease_links = []
        for selector in link_selectors:
            links = browser.find_elements(By.CSS_SELECTOR, selector)
            # Фильтруем пустые href
            valid_links = [l for l in links if l.get_attribute("href") and "/disease/" in l.get_attribute("href")]
            if valid_links:
                disease_links.extend(valid_links)
                break
        
        assert len(disease_links) > 0, "Ссылки на заболевания не найдены"

        first_link = disease_links[0]
        browser.execute_script("arguments[0].click();", first_link)

        time.sleep(3)

        # 4. Проверка страницы деталей
        current_url = browser.current_url
        page_source = browser.page_source.lower()
        
        assert f"{live_server_url}/knowledge-base/" not in current_url or "disease" in current_url, "Не перешли на страницу детализации"
        assert "лечение" in page_source or "описание" in page_source or "симптомы" in page_source, "Нет контента на странице болезни"

    except Exception as e:
        try:
            browser.save_screenshot("e2e_test2_error.png")
        except:
            pass
        pytest.fail(f"E2E тест провален: {e}")

@pytest.mark.e2e
def test_e2e_setup_verification(browser, live_server_url):
    """Тест проверки настроек E2E окружения."""
    assert browser is not None, "Браузер не инициализирован"
    browser.get(live_server_url)
    assert "Медицинский" in browser.page_source, "Главная страница недоступна"
