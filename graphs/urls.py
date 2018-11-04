from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('graphs/', views.graphs, name='graphs',),
    path('main/', views.main, name='main',)
]