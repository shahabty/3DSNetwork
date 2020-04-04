import torch.nn as nn
from torchvision import models
from utils import normalize_imagenet


def get_encoder(model_name,c_dim,normalize = True,use_linear = True):
    if model_name == 'resnet18':
        return Resnet18(c_dim,normalize,use_linear)
    elif model_name == 'resnet34':
        return Resnet34(c_dim,normalize,use_linear)
    elif model_name == 'resnet50':
        return Resnet50(c_dim,normalize,use_linear)
    elif model_name == 'resnet101':
        return Resnet101(c_dim,normalize,use_linear) 
    else:
        raise ValueError('The network name must be resnet* where * could be 18,34,50,101')
class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out

class Resnet34(nn.Module):
    r''' ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out

class Resnet50(nn.Module):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out

class Resnet101(nn.Module):
    r''' ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
                                                
