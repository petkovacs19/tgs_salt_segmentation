from models.unets import resnet34_fpn

def make_model(network, input_shape,channels):  
    if network == 'resnet34':
        return resnet34_fpn(input_shape, channels=channels, activation="softmax") 
    else:
        raise ValueError('unknown network ' + network)