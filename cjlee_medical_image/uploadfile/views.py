from django.shortcuts import render,redirect
from .forms import uploadmodelform
from .models import photo
# Create your views here.
def index(request):
    photos = photo.objects.all()  #查詢所有資料
    form = uploadmodelform()
    if request.method == "POST":
        form = uploadmodelform(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('/uploadfile')
    context = {
        'photos': photos,
        'form': form
    }
    return render(request, 'test2/second.html', context)