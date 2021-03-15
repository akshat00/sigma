from django.db import models
import uuid

# Create your models here.

class UserInfo(models.Model):
    email = models.CharField(max_length = 128, null = False)
    dob = models.DateField(null = False)
    address = models.CharField(max_length = 512, null = False)
    contact = models.CharField(max_length = 10, null = False)

class UploadImage(models.Model): 
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    username = models.CharField(max_length = 150, null=False)
    image = models.ImageField(upload_to = 'images/')
    dateTime = models.DateTimeField(auto_now = True)