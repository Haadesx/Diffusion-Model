import torch
import torch.nn.functional as F

def get_model_fn(model, train=False):

    # hacky fix for now
    def model_fn(x, sigma):
        if train:
            model.train()
        else:
            model.eval()
        
        return model(x, sigma)

    return model_fn

def get_score_fn(model, train=False, sampling=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def score_fn(x, sigma):
            sigma = sigma.reshape(-1)
            score = model_fn(x, sigma)
            
    # idk why but it needs to be this
            if sampling:
                return score.exp()
                
            return score

    return score_fn