from crfseg import MeanFieldCRF
import torch
import torch.nn as nn


def get_crf(dim_in,dim_out,with_crf):
    if with_crf:
        return CRFRNN(dim_in,dim_out)
    else:
        return None

class CRFRNN(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.crf_encoder = nn.Sequential(
        nn.Conv1d(3,3,1),
        nn.LeakyReLU(negative_slope=0.01, inplace=False),
        nn.Conv1d(3,3,1),
        nn.LeakyReLU(negative_slope=0.01, inplace=False),
        nn.Conv1d(3,3,1),
        )
        self.crf = MeanFieldCRF(filter_size = 11,n_iter = 5,return_log_proba=False)

    def forward(self,logits,p):
        #f_out = self.crf_encoder(p.permute(0,2,1))
        return self.crf(logits.unsqueeze(1),p.permute(0,2,1)).squeeze()
