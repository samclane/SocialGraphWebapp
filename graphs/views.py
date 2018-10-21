from io import BytesIO

from django.http import HttpResponse
from django.template import loader
from pylab import *


def graphs(request):
    buf = BytesIO()
    savefig(buf, format='png')
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


def index(request):
    template = loader.get_template('graphs/index.html')
    return HttpResponse(template.render(request=request))

