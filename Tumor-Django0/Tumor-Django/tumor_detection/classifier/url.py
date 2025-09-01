from django.urls import path
from .views import *

urlpatterns = [
    path('', upload_and_classify, name='upload_and_classify'),
]
