import json
import random

import numpy
from django.http import HttpResponse
from django.template import loader
from mpld3 import urls
from mpld3._display import TEMPLATE_DICT
from mpld3.mpld3renderer import MPLD3Renderer
from mpld3.mplexporter import Exporter
from mpld3.utils import get_id
from pylab import re, gcf
from rq import Queue

from .socialgraphing import init_svm_graphs
from worker import conn

metrics = None

# Had to rewrite some lines to let ndarrays through from mpld3._display.py
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def fig_to_html(fig, d3_url=None, mpld3_url=None, no_extras=False,
                template_type="general", figid=None, use_http=False, **kwargs):
    template = TEMPLATE_DICT[template_type]

    d3_url = d3_url or urls.D3_URL
    mpld3_url = mpld3_url or urls.MPLD3_URL

    if use_http:
        d3_url = d3_url.replace('https://', 'http://')
        mpld3_url = mpld3_url.replace('https://', 'http://')

    if figid is None:
        figid = 'fig_' + get_id(fig) + str(int(random.random() * 1E10))
    elif re.search('\s', figid):
        raise ValueError("figid must not contain spaces")

    renderer = MPLD3Renderer()
    Exporter(renderer, close_mpl=False, **kwargs).run(fig)

    fig, figure_json, extra_css, extra_js = renderer.finished_figures[0]

    if no_extras:
        extra_css = ""
        extra_js = ""

    return template.render(figid=json.dumps(figid),
                           d3_url=d3_url,
                           mpld3_url=mpld3_url,
                           figure_json=json.dumps(figure_json, cls=NumpyEncoder),
                           extra_css=extra_css,
                           extra_js=extra_js)
# END UGH


def graphs(request):
    global metrics
    fig = metrics.result[5]
    html = fig_to_html(fig)  # Convert to D3 Graph
    return HttpResponse(html)


def index(request):
    global metrics
    if not metrics:
        q = Queue(connection=conn)
        metrics = q.enqueue(init_svm_graphs, kwargs={'view_percentile': .95})
    if not metrics.is_finished:
        template = loader.get_template('graphs/wrapper.html')
        return HttpResponse(template.render(request=request))
    else:
        template = loader.get_template('graphs/index.html')
        return HttpResponse(template.render(request=request, context={'popularity_list': metrics.result[0],
                                                                      'cross_val': metrics.result[1],
                                                                      'accuracy': metrics.result[2],
                                                                      'class_report': metrics.result[3],
                                                                      'conf_matrix': metrics.result[4]}))

