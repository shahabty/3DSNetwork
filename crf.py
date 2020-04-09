from crfseg import MeanFieldCRF
import torch
import torch.nn as nn


def get_crf(dim_in,dim_out):
    return CRFRNN(dim_in,dim_out)

class CRFRNN(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.crf_encoder = nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.Linear(dim_in,dim_out)
        )
        self.crf = MeanFieldCRF(filter_size = 11,n_iter = 5,return_log_proba=False)

    def forward(self,logits,p):
        f_out = self.crf_encoder(p.permute(0,2,1))
        return self.crf(logits.unsqueeze(1),f_out).squeeze()


