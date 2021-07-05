from django.db import models
from django.contrib import admin
from django.utils import timezone
# Create your models here.
class photo(models.Model):
    image=models.ImageField(upload_to='image/',blank=False,null=False)
    upload_data=models.DateField(default=timezone.now)
