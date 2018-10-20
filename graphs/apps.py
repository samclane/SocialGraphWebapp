from django.apps import AppConfig

from .views import init_svm_graphs


class GraphsConfig(AppConfig):
    name = 'graphs'

    def ready(self):
        print("hello")
        init_svm_graphs()
