import os
import torch
from model import SEDD
import utils
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib

def load_model_local(root_dir, device):
    cfg = utils.load_config_from_run(root_dir)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)

    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise

def load_model(root_dir, device):
    return load_model_local(root_dir, device)
