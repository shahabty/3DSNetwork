from encoder import get_encoder
from decoder import get_decoder
import torch.nn as nn
import torch.distributions as dist


def get_network(cfg, device = None,dataset = None,**kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']

    decoder = get_decoder(dim=dim, z_dim=z_dim, c_dim=c_dim)

        #if z_dim != 0:
        #    encoder_latent = encoder_latent_dict[encoder_latent](dim=dim, z_dim=z_dim, c_dim=c_dim,**encoder_latent_kwargs)
        #else:
        #    encoder_latent = None

        #if encoder == 'idx':
        #    encoder = nn.Embedding(len(dataset), c_dim)
        #elif encoder is not None:
    encoder = get_encoder(model_name = 'resnet18',c_dim=c_dim)
        #else:
        #    encoder = None

        #p0_z = get_prior_z(cfg, device)
    model = OccupancyNetwork(decoder, encoder, device=device)
    return model

class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, encoder_latent=None, p0_z=None,
                 device=None):
        super().__init__()
        #if p0_z is None:
        #    p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder.to(device)

        #if encoder_latent is not None:
       #     self.encoder_latent = encoder_latent.to(device)
       # else:
       #     self.encoder_latent = None

        #if encoder is not None:
        self.encoder = encoder.to(device)
        #else:
        #    self.encoder = None

        self._device = device
        #self.p0_z = p0_z

    def forward(self, p, inputs,sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        c = self.encoder(inputs)
        logits = self.decoder(p, c)

        p_r = dist.Bernoulli(logits=logits)

        return logits,p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


