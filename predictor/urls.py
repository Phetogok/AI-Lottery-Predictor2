from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('draws/', views.draw_list, name='draw_list'),
    path('draws/add/', views.add_draw, name='add_draw'),
    path('draws/import/', views.import_data, name='import_data'),
    path('predictions/', views.prediction_list, name='prediction_list'),
    path('predictions/generate/', views.generate_prediction, name='generate_prediction'),
    path('predictions/<int:pk>/', views.prediction_detail, name='prediction_detail'),
    path('predictions/<int:pk>/refresh/', views.refresh_prediction, name='refresh_prediction'),
    path('statistics/', views.statistics, name='statistics'),
    # Add this URL pattern to your urlpatterns list
    path('predictions/clear-all/', views.clear_all_predictions, name='clear_all_predictions'),
]