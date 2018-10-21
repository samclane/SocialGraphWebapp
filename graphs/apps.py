from django.apps import AppConfig

from .socialgraphing import init_svm_graphs


class GraphsConfig(AppConfig):
    name = 'graphs'

    def ready(self):
        print("Initializing app...")
        init_svm_graphs()
