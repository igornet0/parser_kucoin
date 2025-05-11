import importlib

class GUICheck:
    @classmethod
    def has_gui_deps(cls):
        try:
            importlib.import_module('selenium')
            return True
        except ImportError:
            return False
