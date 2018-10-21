from io import BytesIO

from django.http import HttpResponse
from pylab import *


def index(request):
    buf = BytesIO()
    savefig(buf, format='png')

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response
