from django.shortcuts import render_to_response
def index(request):
    content={}
    return render_to_response('home.html',content)
def search(requset):
    content = {}
    content['org'] = ""
    if requset.POST:
        tmp = requset.POST['q']
        content['org'] = tmp
        content['result'] = tmp
    return render_to_response('search.html',content)
