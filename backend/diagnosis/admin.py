from django.contrib import admin
from .models import Disease, Symptom

@admin.register(Disease)
class DiseaseAdmin(admin.ModelAdmin):
    """
    Настройки отображения и взаимодействия с моделью Disease в админ-панели.
    """
    # Поля, которые будут отображаться в списке заболеваний
    list_display = ('name', 'severity', 'category', 'specialist', 'id')
    
    # Поля для фильтрации списка справа
    list_filter = ('severity', 'category', 'specialist')
    
    # Поля, по которым можно выполнять поиск (будет добавлено поле поиска)
    search_fields = ('name', 'description', 'treatment')
    
    # Поля, которые будут отображаться как ссылки для перехода к редактированию
    list_display_links = ('name', 'id')

    # Определение полей для отображения в детальном представлении (редактирование)
    fieldsets = (
        (None, {
            'fields': ('name', 'description', 'treatment', 'symptoms'),
        }),
        ('Классификация', {
            'fields': ('severity', 'category', 'specialist'),
            'classes': ('collapse',),
        }),
    )

@admin.register(Symptom)
class SymptomAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)