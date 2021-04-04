"""sigma URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from webapp.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name = 'home'),
    path('login/', login_view, name = 'login'),
    path('register/', register_view, name = 'register'),
    path('overview/', overview_view, name = 'overview'),
    path('about/', about_view, name = 'about'),
    path('logout/', logout_view, name = 'logout'),
    path('upload/', upload_view, name = 'upload-images'),
    path('low_light/', low_light_view, name = 'low-light'),
    path('dehazing/', dehaze_view, name = 'dehaze'),
    path('black_white/', black_white_view, name = 'black-white')

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)