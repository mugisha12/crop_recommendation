
from django.urls import path
from . import views # Import views from the current app

app_name = 'recommender_app' # Namespace for your app's URLs

urlpatterns = [
    path('', views.index, name='predict_crop_index'),
    path('predict', views.predict_crop, name='predict_crop'),
]