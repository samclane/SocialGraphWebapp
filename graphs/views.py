from django.http import HttpResponse
from pylab import plot, xlabel, ylabel, title, grid, savefig, plt
from io import BytesIO
from matplotlib.figure import Figure
import numpy as np

def index(request):
    # Construct the graph
    fig = Figure()
    ax = fig.add_subplot(111)
    x = np.arange(0, 2*pi, 0.01)
    s = cos(x)**2
    ax.plot(x, s)

    plot(x, s)

    xlabel('xlabel(X)')
    ylabel('ylabel(Y)')
    title('Simple Graph!')
    grid(True)

    buf = BytesIO()
    savefig(buf, format='png')
    plt.close(fig)

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response