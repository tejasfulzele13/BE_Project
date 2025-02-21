from django.urls import path

from . import views

urlpatterns=[
    path('', views.home,name ="home"),
    path("classify/", views.classify_attack, name="classify_traffic"),
    path('authenticate/', views.authenticate_user, name='authenticate'),
    path("get_traffic_logs/", views.get_traffic_logs, name="get_traffic_logs"),
    path("dashboard/", views.traffic_dashboard, name="traffic_dashboard"),
     path("clear_logs/", views.clear_logs, name="clear_logs"),
     path('train-models/', views.train_models, name='train_models'),
    path('results/', views.results_page, name='results_page'),

]


