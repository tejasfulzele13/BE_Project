from django.urls import path

from . import views

urlpatterns=[
    path('', views.home,name ="home"),
    path("classify/", views.classify_attack, name="classify_traffic"),
     path('authenticate/', views.authenticate_user, name='authenticate'),
    path('dashboard/', views.dashboard, name='dashboard'),

]


