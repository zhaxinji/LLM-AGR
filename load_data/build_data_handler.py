
from config.configurator import configs
import importlib

def build_data_handler():
    handler_name = f"data_handler_{configs['data']['type']}"
    module_path = f"load_data.{handler_name}"
    if importlib.util.find_spec(module_path) is None:
        raise NotImplementedError(f"DataHandler {handler_name} is not implemented")
    module = importlib.import_module(module_path)
    handler_class = next((getattr(module, attr) for attr in dir(module) if attr.lower() == handler_name.lower().replace('_', '')), None)
    if handler_class is None:
        raise NotImplementedError(f"DataHandler Class {handler_name} is not defined in {module_path}")
    return handler_class()