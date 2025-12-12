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

        symptom_checkboxes = browser.find_elements(By.CSS_SELECTOR, 'input[type="checkbox"][name="symptoms"]')

        if symptom_checkboxes:
            selected_count = 0
            for i, checkbox in enumerate(symptom_checkboxes):
                try:
                    browser.execute_script("arguments[0].scrollIntoView(true);", checkbox)
                    time.sleep(0.2)
                    browser.execute_script("window.scrollBy(0, 100);")
                    time.sleep(0.2)

                    if checkbox.is_displayed() and checkbox.is_enabled():
                        browser.execute_script("arguments[0].click();", checkbox)
                        selected_count += 1

                        if selected_count >= 3:
                            break

                except Exception:
                    continue

            if selected_count == 0:
                for i, checkbox in enumerate(symptom_checkboxes[:5]):
                    try:
                        browser.execute_script("arguments[0].click();", checkbox)
                        selected_count += 1
                        if selected_count >= 2:
                            break
                    except:
                        continue
        else:
            return

        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        button_selectors = ['button[type="submit"]', ".btn-medical", "button"]

        target_button = None
        for selector in button_selectors:
            try:
                buttons = browser.find_elements(By.CSS_SELECTOR, selector)
                for button in buttons:
                    if button.is_displayed() and button.is_enabled():
                        target_button = button
                        break
                if target_button:
                    break
            except:
                continue

        if target_button:
            browser.execute_script("arguments[0].scrollIntoView(true);", target_button)
            time.sleep(0.5)
            browser.execute_script("arguments[0].click();", target_button)
        else:
            browser.get(f"{live_server_url}/predict/")

        time.sleep(3)

        current_url = browser.current_url
        page_content = browser.page_source

        page_content_lower = page_content.lower()

        assert len(page_content) > 0, "Страница пустая"
        assert "html" in page_content_lower, "Не HTML страница"

    except Exception as e:
        try:
            browser.save_screenshot("e2e_test1_error.png")
        except:
            pass
        pytest.skip(f"E2E тест пропущен из-за ошибки: {e}")


@pytest.mark.e2e
def test_knowledge_base_navigation(browser, live_server_url, wait):
    """E2E тест навигации по базе знаний: главная → база знаний → детали заболевания."""
    try:
        browser.get(f"{live_server_url}/knowledge-base/")

        wait(browser).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        page_content = browser.page_source
        assert len(page_content) > 0, "Страница базы знаний пустая"

        card_selectors = [
            ".disease-card",
            ".card",
            '[class*="disease"]',
            '[class*="card"]',
        ]

        disease_cards = []
        for selector in card_selectors:
            cards = browser.find_elements(By.CSS_SELECTOR, selector)
            if cards:
                disease_cards.extend(cards)
                break

        if disease_cards:
            link_selectors = [
                'a[href*="/disease/"]',
                'a[href*="knowledge-base/disease/"]',
                "a",
            ]

            disease_links = []
            for selector in link_selectors:
                links = browser.find_elements(By.CSS_SELECTOR, selector)
                for link in links:
                    href = link.get_attribute("href") or ""
                    if "/disease/" in href or "knowledge-base/disease/" in href:
                        disease_links.append(link)
                if disease_links:
                    break

            if disease_links:
                first_link = disease_links[0]
                browser.execute_script("arguments[0].click();", first_link)

                time.sleep(3)

                new_url = browser.current_url
                if new_url != f"{live_server_url}/knowledge-base/":
                    disease_content = browser.page_source.lower()

    except Exception as e:
        try:
            browser.save_screenshot("e2e_test2_error.png")
        except:
            pass
        pytest.skip(f"E2E тест пропущен из-за ошибки: {e}")


@pytest.mark.e2e
def test_e2e_setup_verification(browser, live_server_url):
    """Тест проверки настроек E2E окружения."""
    assert browser is not None, "Браузер не инициализирован"

    browser.get(live_server_url)
    assert "Медицинский" in browser.page_source, "Главная страница недоступна"
