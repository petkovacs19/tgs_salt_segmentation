from models.unets import resnet34_fpn

def make_model(network, input_shape):  
    if network == 'resnet34':
        return resnet34_fpn(input_shape, channels=2, activation="softmax") 
    else:
        raise ValueError('unknown network ' + network)