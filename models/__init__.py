import os
import torch
from torch import nn
import importlib

class ModelSaver:
    """Keep last n checkpoints. If keep_last_ckpts is None, keep all."""

    def __init__(self, model, keep_last_ckpts=None):
        self.ckpts = []
        self.model = model
        self.keep_last_ckpts = keep_last_ckpts

    def save(self, destination, **additional_infos):
        self.ckpts.append(destination)
        self.model.save(destination, **additional_infos)
        if self.keep_last_ckpts and len(self.ckpts) > self.keep_last_ckpts:
            print("Deleting checkpoint: %s" % self.ckpts[0])
            os.remove(self.ckpts[0])
            self.ckpts = self.ckpts[1:]

class ModelFactory:
    """Very lightweight model creation."""

    @staticmethod
    def build_model(arch, args):
        # from . import MODEL_REGISTRY
        return MODEL_REGISTRY[arch].build_model(args)

class BaseModel(nn.Module):
    """Very lightweight model definition."""

    NAME = None

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args

    def save(self, output_path, **additional_infos):
        """Internally checks the --no-save-model, if True, does not save."""
        if self.args.no_save_model:
            return
        state = dict(args=self.args, class_name=self.NAME, state_dict=self.state_dict())
        state.update(additional_infos)
        torch.save(state, output_path)

    def forward(self, x, target, **kwargs):
        """Compute training costs. Should return pre-softmax values
        and a dict with a ``main_cost'' key.
        """
        raise NotImplementedError

    @staticmethod
    def add_args():
        return

    @staticmethod
    def build_model(args):
        raise NotImplementedError

    @staticmethod
    def load(input_path):
        """load model given input path"""
        state = torch.load(input_path)
        print("Loading model: %s" % state["class_name"])
        model = MODEL_REGISTRY[state["class_name"]](state["args"])
        model.load_state_dict(state["state_dict"])
        return model

MODEL_REGISTRY = {}


def register_model(name):
    """Registers all models in a model registry
    """
    def register_model_cls(cls):
        cls.NAME = name
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        print("Registering new model %s." % name)
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls

"""Import model modules
"""
for file in os.listdir(os.path.dirname(__file__)):
    if file.startswith("model_"):
        module = importlib.import_module("." + file[:-3], package="models")
