from models.unets import resnet34_fpn, simple_resnet

def make_model(network, input_shape, channels):  
    if network == 'resnet34':
        return resnet34_fpn(input_shape, channels=channels, activation="softmax")
    if network == 'simple_resnet':
        return simple_resnet(input_shape, channels, 0.5)
    else:
        raise ValueError('unknown network ' + network)