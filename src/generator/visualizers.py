import torch
from torch.nn import Module
from torchview import draw_graph


def model_params_count(module: Module) -> int:
    count = sum(p.numel() for p in module.parameters())
    return count


def model_input_shape(module: Module) -> tuple:
    return list(module.parameters())[0].shape


def draw_model(module: Module, filename: str, folder: str, depth: int = 8, **kwargs):
    shape_of_first_layer = list(module.parameters())[0].shape
    model_graph = draw_graph(
        module,
        input_size=shape_of_first_layer,
        graph_name=filename,
        graph_dir="LR",
        depth=depth,
        hide_inner_tensors=True,
        hide_module_functions=True,
        save_graph=True,
        expand_nested=True,
        show_shapes=True,
        filename=filename,
        directory=folder,
    )
    return model_graph
