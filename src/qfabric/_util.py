import importlib


def dynamic_import(module_path: str, class_name: str):
    """Import a class via module path and class name."""
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls
