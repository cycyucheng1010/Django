from django.shortcuts import render,redirect
from .forms import uploadmodelform
from .models import photo
# Create your views here.
def index(request):
    #pic_obj = models.photo.objects.clear()
    # photos = photo.objects.all()  #查詢所有資料
    preview=photo.objects.last() # 查詢最後一筆資料
    form = uploadmodelform()
    if request.method == "POST":
        form = uploadmodelform(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    context = {
        #'photos': photos,
        'form': form,
        'preview':preview
    }
    return render(request, 'test2/second.html', context)    

