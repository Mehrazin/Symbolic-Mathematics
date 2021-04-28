from .transformer import *


def build_model(config) :
    """
    builds or reloads model and transfer them to config.device
    """

    model = {}
    # model['encoder'] =
    # model['decoder'] =

    if config.load_model:
        checkpoint = torch.load(config.model_path)
        print("=> Loading checkpoint")
        for k in list(model.keys()) :
            assert k in lsit(checkpoint.keys())
            model[k].load_state_dict(checkpoint[k])
        print('Model loaded')
    for k in list(model.keys()) :
        model[k].to(config.device)
    return model
