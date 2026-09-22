from .config import load_config

__all__ = ["load_config", "ProsodicLLM"]


def __getattr__(name):
    if name == "ProsodicLLM":
        from .model import ProsodicLLM
        return ProsodicLLM
    raise AttributeError(name)
