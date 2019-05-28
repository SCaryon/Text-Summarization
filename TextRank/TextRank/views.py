from django.shortcuts import render_to_response
def index(request):
    content={}
    return render_to_response('home.html',content)
def others(request):
    content={}
    return render_to_response('others.html',content)