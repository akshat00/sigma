from django.db import models

# Create your models here.

class UserInfo(models.Model):
    email = models.CharField(max_length = 128, null = False)
    dob = models.DateField(null = False)
    address = models.CharField(max_length = 512, null = False)
    contact = models.CharField(max_length = 10, null = False)