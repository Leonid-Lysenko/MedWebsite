from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('disease/<str:disease_name>/', views.disease_detail, name='disease_detail'),
    path('about/', views.about, name='about'),
    path('how-to-use/', views.how_to_use, name='how_to_use'),
]