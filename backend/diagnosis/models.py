from django.db import models

class Disease(models.Model):
    """Модель для хранения детальной информации о заболевании."""
    
    # Название заболевания - должно быть уникальным
    name = models.CharField(max_length=255, unique=True, verbose_name="Название заболевания")
    
    # Краткое описание
    description = models.TextField(verbose_name="Описание")
    
    # Рекомендации по лечению
    treatment = models.TextField(verbose_name="Лечение")
    
    # Список симптомов (в виде JSON-массива строк)
    symptoms = models.JSONField(verbose_name="Симптомы") 
    
    # Степень серьезности ('high', 'medium', 'low', 'unknown')
    severity = models.CharField(max_length=50, verbose_name="Степень серьезности")
    
    # Специалист
    specialist = models.CharField(max_length=100, verbose_name="Специалист")
    
    # Категория
    category = models.CharField(max_length=100, verbose_name="Категория")

    class Meta:
        verbose_name = "Заболевание"
        verbose_name_plural = "Заболевания"

    def __str__(self):
        return self.name