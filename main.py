from network import get_network
from data_loader import get_dataset
from utils import *
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from checkpoints import CheckpointIO
logger = SummaryWriter(os.path.join('out', 'logs')) 


cfg = {
'data':{'dataset': 'Shapes3D',
  'path': '/media/shahab/D2/CV-project/3DSNetwork/data/ShapeNet/',
  'classes': None,
  'input_type': 'img',
  'img_folder':'img_choy2016',
  'train_split': 'train',
  'val_split': 'val',
  'test_split': 'test',
  'dim': 3,
  'points_file': 'points.npz',
  'points_iou_file': 'points.npz',
  'points_subsample': 2048,
  'points_unpackbits': 'true',
  'model_file': 'model.off',
  'watertight_file': 'model_watertight.off',
  'img_size': 224,
  'img_with_camera': False,
  'img_augment': False,
  'n_views': 24,
  'pointcloud_file': 'pointcloud.npz',
  'pointcloud_chamfer_file': 'pointcloud.npz',
  'pointcloud_n': 256,
  'pointcloud_target_n': 1024,
  'pointcloud_noise': 0.05,
  'voxels_file': 'model.binvox',
  'with_transforms': False,
},
'model':{'c_dim': 256,'z_dim': 0},
'train':{'batch_size': 64,'epochs':20,'pretrained':'onet_img2mesh_3-f786b04a.pt'},
'out': {'out_dir':'out','checkpoint_dir':'pretrained'}
}


def main(cfg):
    train_dataset = get_dataset(mode = 'train',cfg = cfg)
    val_dataset = get_dataset(mode = 'val',cfg = cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], num_workers=8, shuffle=True, collate_fn=collate_remove_none,worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, num_workers=8, shuffle=False,collate_fn=collate_remove_none,worker_init_fn=worker_init_fn)

    model = get_network(cfg,device = 'cuda:0',dataset = train_dataset)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    checkpoint = CheckpointIO(cfg['out']['checkpoint_dir'], model=model, optimizer=optimizer) 
    load_dict = checkpoint.load(cfg['train']['pretrained'])


    train(train_loader,val_loader,model,optimizer,checkpoint,cfg)

    
def train(train_loader,val_loader,model,optimizer,checkpoint,cfg):

    for epoch in range(cfg['train']['epochs']):
        model.train()
        i = 0
        for batch in train_loader:
            p = batch.get('points').to('cuda:0')
            occ = batch.get('points.occ').to('cuda:0')
            inputs = batch.get('inputs',torch.empty(p.size(0),0)).to('cuda:0')
            optimizer.zero_grad()
            logits = model(p,inputs)

            # General points
            loss = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
            loss = loss.sum(-1).mean()
            loss.backward()
            optimizer.step()
            i+=1
            print('iteration:{}/480'.format(i)) 
                  


def validation(val_loader,model,optimizer,checkpoint,cfg):
        model.eval()

        threshold = self.threshold
        eval_dict = {}
        for data in val_loader:
            # Compute elbo
            points = data.get('points').to('cuda:0')
            occ = data.get('points.occ').to('cuda:0')

            inputs = data.get('inputs', torch.empty(points.size(0), 0)).to('cuda:0')
            voxels_occ = data.get('voxels')

            points_iou = data.get('points_iou').to('cuda:0')
            occ_iou = data.get('points_iou.occ').to('cuda:0')


            with torch.no_grad():
                elbo, rec_error, kl = self.model.compute_elbo(points, occ, inputs, **kwargs)

            eval_dict['loss'] = -elbo.mean().item()
            eval_dict['rec_error'] = rec_error.mean().item()
            eval_dict['kl'] = kl.mean().item()

            # Compute iou
            batch_size = points.size(0)

            with torch.no_grad():
                p_out = model(points_iou, inputs,sample=self.eval_sample, **kwargs)

            occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
            occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
            eval_dict['iou'] = iou

            # Estimate voxel iou
            if voxels_occ is not None:
                voxels_occ = voxels_occ.to(device)
                points_voxels = make_3d_grid((-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
                points_voxels = points_voxels.expand(batch_size, *points_voxels.size())
                points_voxels = points_voxels.to(device)
                with torch.no_grad():
                    p_out = self.model(points_voxels, inputs,sample=self.eval_sample, **kwargs)

                voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
                occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
                iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

                eval_dict['iou_voxels'] = iou_voxels

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        print(eval_dict)


main(cfg) 

