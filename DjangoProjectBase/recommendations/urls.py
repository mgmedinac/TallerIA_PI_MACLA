from django.urls import path
from . import views 

urlpatterns = [
    path('', views.recommend, name='recommendations'), # pagina principal de recommendations
]