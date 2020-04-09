from encoder import get_encoder
from decoder import get_decoder
import torch.nn as nn
import torch.distributions as dist
from crf import get_crf


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
    encoder = get_encoder(model_name = 'resnet18',c_dim=c_dim)
    crf = get_crf(dim_in = cfg['data']['points_subsample'],dim_out = cfg['data']['points_subsample'])
    model = OccupancyNetwork(decoder, encoder,crf,device=device)
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

    def __init__(self, decoder, encoder=None,crf = None,device=None):
        super().__init__()
        self.decoder = decoder.to(device)
        self.encoder = encoder.to(device)
        if crf is not None: 
            self.crf = crf.to(device)
        self._device = device

    def forward(self, p, inputs,sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
#        print(p.shape)
        batch_size = p.size(0)
        c = self.encoder(inputs)
        logits = self.decoder(p, c)
        crf_out = None
        if self.crf != None:
            crf_out = self.crf(logits,p)
            #crf_out = self.crf(logits.unsqueeze(1),p.permute(0,2,1)).squeeze()
        p_r = dist.Bernoulli(logits=crf_out)#logits)

#        return logits,p_r,crf_out
        return crf_out,p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


