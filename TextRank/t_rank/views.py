from django.shortcuts import render,render_to_response
# Create your views here.
def basic(request):
    content={}
    return render_to_response('t_rank.html',content)

def res(renquest):
    content = {}
    content['org'] = ""
    if renquest.POST:
        tmp = renquest.POST['q']
        content['org'] = tmp
        content['result'] = tmp
    return render_to_response('search.html',content)