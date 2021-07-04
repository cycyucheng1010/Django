from django.shortcuts import render
from .models import Vendor
from .forms import VendorForm #tips: remember import correspond Model Form!
# Create your views here.
def vendor_index(request):
    vendor_list = Vendor.objects.all() 
    context = {'vendor_list':vendor_list}
    #render will catch the file which in templates.
    return render(request,'vendor/vendor_detail.html',context)

def vendor_create_view(request):
    form = VendorForm(request.POST or None)
    if form.is_valid(): # form.is_valid: verify the data is correct
        form.save()
        form = VendorForm() # clear form
    context={'form':form}
    return render(request,'vendor/vendor_create.html',context)