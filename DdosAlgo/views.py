from django.shortcuts import render
import requests

# Create your views here.
def home(requests):
    return render(requests , 'home.html')

