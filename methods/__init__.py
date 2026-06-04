import importlib


class LazyMethod:
    def __init__(self, module_name, function_name):
        self.module_name = module_name
        self.function_name = function_name
        self.__name__ = function_name

    def _load(self):
        return getattr(importlib.import_module(self.module_name), self.function_name)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __repr__(self):
        return f"<lazy method {self.module_name}.{self.function_name}>"


base = LazyMethod("methods.base", "base")
tool_augmentation = LazyMethod("methods.tool_augmentation", "tool_augmentation")
self_correction = LazyMethod("methods.self_correction", "self_correction")
self_consistency = LazyMethod("methods.self_consistency", "self_consistency")

__all__ = [
    "base",
    "tool_augmentation",
    "self_correction",
    "self_consistency",
]
