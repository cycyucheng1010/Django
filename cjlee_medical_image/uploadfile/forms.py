from django import forms
from .models import photo

class uploadmodelform(forms.ModelForm):
    class Meta:
        model = photo
        fields =('image',)
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control-file'})
        }