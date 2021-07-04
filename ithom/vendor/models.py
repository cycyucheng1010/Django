from django.db import models
from django.contrib import admin
# Create your models here.
class Vendor(models.Model):
    # set vendor name, store name, phone number, address
    vendor_name = models.CharField(max_length = 20)
    store_name = models.CharField(max_length = 10)
    phone_number = models.CharField(max_length = 20)
    address = models.CharField(max_length = 100)

    def __str__(self):
        return self.vendor_name

class Food(models.Model):
    # set food name, food price 
    food_name = models.CharField(max_length = 30)
    price_name = models.DecimalField(max_digits = 3, decimal_places = 0) # max_digital: the length before decimal point, decimal_places:the length after decimal point
    #food is made by which vendor
    food_vendor = models.ForeignKey(Vendor,on_delete = models.CASCADE) # on_dekete: when the match class has been deleted ,CASCADE: all delete

@admin.register(Vendor)
class VenderAdmin(admin.ModelAdmin):
    list_display = ['id','vendor_name','vendor_name','store_name','phone_number','address']

@admin.register(Food)
class FoodAdmin(admin.ModelAdmin):
    list_display=[field.name for field in Food._meta.fields]
    list_filter=('price_name',)
    