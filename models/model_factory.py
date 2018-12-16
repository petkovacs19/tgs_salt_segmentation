from models.unets import resnet34_fpn
from models.unets import build_model

def make_model(network, input_shape, channels, activation="softmax"):  
    if network == 'resnet34':
        return resnet34_fpn(input_shape)
    elif network == 'custom_resnet':
        return build_model(input_shape)
    else:
        raise ValueError('unknown network ' + network)