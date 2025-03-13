# ml_tool/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_dataset, name='home'),  # Root URL
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('select-models/', views.select_models, name='select_models'),
    path('perform-voting/', views.perform_voting, name='perform_voting'),
    path('delete_dataset/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),
]