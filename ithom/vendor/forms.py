from django import forms
from .models import Vendor,Food

class VendorForm(forms.ModelForm):
    class Meta:
        model = Vendor # which model we choose
        fields = '__all__' # in this model which field we choose