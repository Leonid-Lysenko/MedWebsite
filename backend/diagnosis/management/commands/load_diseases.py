# MedWebsite/backend/diagnosis/management/commands/load_diseases.py

from django.core.management.base import BaseCommand
from diagnosis.models import Disease
# Временно импортируем старый файл, чтобы получить данные
try:
    from diagnosis.disease_data import DISEASE_DATABASE 
except ImportError:
    # Защита на случай, если файл будет перемещен или удален до запуска
    DISEASE_DATABASE = {}
    
class Command(BaseCommand):
    help = 'Загружает данные о заболеваниях из disease_data.py в базу данных.'

    def handle(self, *args, **options):
        
        if not DISEASE_DATABASE:
            self.stdout.write(self.style.ERROR('Ошибка: DISEASE_DATABASE пуст. Проверьте путь к disease_data.py.'))
            return

        # Удаляем старые данные, чтобы избежать дублирования при повторном запуске
        deleted_count, _ = Disease.objects.all().delete()
        
        diseases_to_create = []
        count = 0

        self.stdout.write(self.style.SUCCESS(f'Начало загрузки данных о заболеваниях. Удалено {deleted_count} старых записей.'))

        for name, data in DISEASE_DATABASE.items():
            diseases_to_create.append(
                Disease(
                    name=name,
                    description=data.get("description", "Описание отсутствует"),
                    treatment=data.get("treatment", "Лечение не указано"),
                    symptoms=data.get("symptoms", []), # JSONField принимает список
                    severity=data.get("severity", "unknown"),
                    specialist=data.get("specialist", "Терапевт"),
                    category=data.get("category", "Уточняется"),
                )
            )
            count += 1
            
        # Массовое создание объектов для повышения производительности
        Disease.objects.bulk_create(diseases_to_create)
        
        self.stdout.write(self.style.SUCCESS(f'Успешно загружено {count} записей о заболеваниях.'))