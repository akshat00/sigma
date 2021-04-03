from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from django.http import HttpResponse
from main.test import low_light_enhancement
import os
from src.example import dehaze_image

from .models import UserInfo, UploadImage
from .forms import UploadImageForm


# Create your views here.

def home_view(request):
    context = {}
    if(request.user.is_authenticated):
        images = UploadImage.objects.filter(username=request.user)

        context['images'] = images

        return render(request, "home.html", context)

    else:
        return redirect(login_view)


def login_view(request):
    context = {}
    if(request.method == 'POST'):
        email = request.POST.get('email')
        password = request.POST.get('password')
        username = User.objects.filter(email=email)

        if(username.exists()):
            user = authenticate(
                request, username=username[0], password=password)

            if user is not None:
                login(request, user)
                return redirect(home_view)

            else:
                context['error'] = ['Invalid Password']

        else:
            context['error'] = ['Invalid Username']

        return render(request, "login/login.html", context)

    else:
        return render(request, "login/login.html", context)


def register_view(request, *args, **kwargs):
    context = {}

    if(request.method == 'POST'):
        if(request.POST.get('password1') == request.POST.get('password2')):
            if(not User.objects.filter(username=request.POST.get('username')).exists()):
                if(len(request.POST.get('contact')) == 10):
                    fname = request.POST.get('fname')
                    lname = request.POST.get('lname')
                    dob = request.POST.get('dob')
                    address = request.POST.get('address')
                    contact = request.POST.get('contact')
                    username = request.POST.get('username')
                    email = request.POST.get('email')
                    password = request.POST.get('password1')

                    user = User.objects.create_user(
                        first_name=fname, last_name=lname, email=email, username=username, password=password)
                    user.save()

                    user_info = UserInfo(
                        email=email, dob=dob, address=address, contact=contact)
                    user_info.save()

                else:
                    context['error'] = ['The contact number is invalid.']

            else:
                context['error'] = ["The username is already taken"]

        else:
            context['error'] = ["Passwords entered do not match."]

        return render(request, "login/register.html", context)

    else:
        return render(request, "login/register.html", context)


def overview_view(request):
    return render(request, 'login/overview.html', {})


def about_view(request):
    return render(request, 'login/about.html', {})


def logout_view(request):
    logout(request)
    return redirect(home_view)


def upload_view(request):
    imageForm = UploadImageForm()
    context = {}
    context['form'] = imageForm
    user = str(request.user)
    context['username'] = user

    if request.method == 'POST':
        imageForm = UploadImageForm(request.POST, request.FILES)

        if imageForm.is_valid():
            imageForm.save()

    return render(request, "upload_image.html", context)

def low_light_view(request):
    base_url = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_url = base_url + request.POST.get('file_name')

    low_light_enhancement(image_url, base_url)

    return redirect(home_view)

def dehaze_view(request):
    base_url = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_url = base_url + request.POST.get('file_name')

    dehaze_image(image_url, base_url)

    return redirect(home_view)
