from django.apps import AppConfig

from .socialgraphing import init_svm_graphs

metrics = ()

class GraphsConfig(AppConfig):
    name = 'graphs'

    def ready(self):
        print("Initializing app...")
        global metrics
        metrics = init_svm_graphs(view_percentile=0.90)
