from network import get_network
from data_loader import get_dataset
from generation import get_generator
from utils import *
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from checkpoints import CheckpointIO
from tqdm import tqdm
from collections import defaultdict
import shutil 


cfg = {
'mode':'train',
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
  'points_unpackbits': True,
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
'val':{'batch_size':10},
'test':{'pretrained':'onet_img2mesh_3-f786b04a.pt','vis_n_outputs': 30},
'out': {'out_dir':'out_crf','checkpoint_dir':'pretrained','save_freq':5}
}

logger = SummaryWriter(os.path.join(cfg['out']['out_dir'], 'logs'))

def main(cfg):

    if cfg['mode'] == 'train':
        train_dataset = get_dataset(mode = 'train',cfg = cfg)
        val_dataset = get_dataset(mode = 'val',cfg = cfg)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], num_workers=8, shuffle=True, collate_fn=collate_remove_none,worker_init_fn=worker_init_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['val']['batch_size'], num_workers=8, shuffle=False,collate_fn=collate_remove_none,worker_init_fn=worker_init_fn)
        model = get_network(cfg,device = 'cuda:0',dataset = train_dataset)      
    else:
        test_dataset = get_dataset(mode = 'test', cfg = cfg, return_idx=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)
        model = get_network(cfg,device = 'cuda:0',dataset = test_dataset)

    if cfg['mode'] == 'train':
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    else:
        optimizer = None 

    if cfg['mode'] == 'train':
        checkpoint = CheckpointIO(cfg['out']['checkpoint_dir'], model=model, optimizer=optimizer)
        load_dict = checkpoint.load(cfg['train']['pretrained'])
        train(train_loader,val_loader,model,optimizer,checkpoint,cfg)
    else:
        checkpoint = CheckpointIO(cfg['out']['checkpoint_dir'], model=model)
        load_dict = checkpoint.load(cfg['test']['pretrained']) 
        test(test_loader,test_dataset,model,cfg)
    
def train(train_loader,val_loader,model,optimizer,checkpoint,cfg):
    it = 0
    for epoch in range(cfg['train']['epochs']):
        model.train()
        #validation(val_loader,model,optimizer,checkpoint,cfg,it)
        for batch in tqdm(train_loader):
            p = batch.get('points').to('cuda:0')
            occ = batch.get('points.occ').to('cuda:0')
            inputs = batch.get('inputs',torch.empty(p.size(0),0)).to('cuda:0')
            optimizer.zero_grad()
            logits,p_r = model(p,inputs)

            # General points
            loss = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
            loss = loss.sum(-1).mean()
            loss.backward()
            optimizer.step()
            
            logger.add_scalar('train loss', loss, it)
            it+=1

        checkpoint.save('model_%s.pt'%str(epoch), epoch_it=epoch, it=it)
        validation(val_loader,model,optimizer,checkpoint,cfg,it)


def validation(val_loader,model,optimizer,checkpoint,cfg,it):
        model.eval()

        threshold = 0.5
        mean_iou = 0.0
        mean_rec_error = 0.0
        mean_iou_voxels = 0.0
        for data in tqdm(val_loader):
            points = data.get('points').to('cuda:0')
            occ = data.get('points.occ').to('cuda:0')

            inputs = data.get('inputs', torch.empty(points.size(0), 0)).to('cuda:0')
            voxels_occ = data.get('voxels')

            points_iou = data.get('points_iou').to('cuda:0')
            occ_iou = data.get('points_iou.occ').to('cuda:0')


            with torch.no_grad():
                logits, p_r = model(points, inputs)
                rec_error = -p_r.log_prob(occ).sum(dim=-1)
 
            mean_rec_error += rec_error.mean().item()

            # Compute iou
            batch_size = points.size(0)

            with torch.no_grad():
                logits,p_out = model(points_iou, inputs,sample=False)

            occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
            occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
            mean_iou += iou

            # Estimate voxel iou
            if voxels_occ is not None:
                voxels_occ = voxels_occ.to('cuda:0')
                points_voxels = make_3d_grid((-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
                points_voxels = points_voxels.expand(batch_size, *points_voxels.size())
                points_voxels = points_voxels.to('cuda:0')
                with torch.no_grad():
                    logits, p_out = model(points_voxels, inputs,sample=False)

                voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
                occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
                iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

                mean_iou_voxels += iou_voxels

        logger.add_scalar('val reconstruction error:', mean_rec_error/len(val_loader), it)
        logger.add_scalar('val points iou:', mean_iou/len(val_loader), it)
        logger.add_scalar('val voxels iou', mean_iou_voxels/len(val_loader), it)



def test(test_loader,test_dataset,model,cfg):
    model.eval()
    generator = get_generator(model)
    model_counter = defaultdict(int)
    for it,data in enumerate(tqdm(test_loader)):
        # Output folders
        mesh_dir = os.path.join(cfg['out']['out_dir'], 'meshes')
        pointcloud_dir = os.path.join(cfg['out']['out_dir'], 'pointcloud')
        in_dir = os.path.join(cfg['out']['out_dir'], 'input')
        generation_vis_dir = os.path.join(cfg['out']['out_dir'], 'vis', )

        # Get index etc.
        idx = data['idx'].item()

        try:
            model_dict = test_dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}

        modelname = model_dict['model']
        category_id = model_dict.get('category', 'n/a')

        try:
            category_name = test_dataset.metadata[category_id].get('name', 'n/a')
        except AttributeError:
            category_name = 'n/a'

        if category_id != 'n/a':
            mesh_dir = os.path.join(mesh_dir, str(category_id))
            pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
            in_dir = os.path.join(in_dir, str(category_id))

            folder_name = str(category_id)
            if category_name != 'n/a':
                folder_name = str(folder_name) + '_' + category_name.split(',')[0]

            generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

        if  not os.path.exists(generation_vis_dir):
            os.makedirs(generation_vis_dir)

        if  not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)

        if  not os.path.exists(pointcloud_dir):
            os.makedirs(pointcloud_dir)

        if not os.path.exists(in_dir):
            os.makedirs(in_dir)

        # Timing dict
        #time_dict = {
        #    'idx': idx,
        #    'class id': category_id,
        #    'class name': category_name,
        #    'modelname': modelname,
        #}
        #time_dicts.append(time_dict)

        # Generate outputs
        out_file_dict = {}

        # Also copy ground truth
#        modelpath = os.path.join(test_dataset.dataset_folder, category_id, modelname,'model_watertight.off')
#        out_file_dict['gt'] = modelpath

    #if generate_mesh:
        #t0 = time.time()
        out = generator.generate_mesh(data)
        #time_dict['mesh'] = time.time() - t0

        # Get statistics
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        #time_dict.update(stats_dict)

        # Write output
        mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
        mesh.export(mesh_out_file)
        out_file_dict['mesh'] = mesh_out_file

    #if generate_pointcloud:
    #    t0 = time.time()
    #    pointcloud = generator.generate_pointcloud(data)
    #    time_dict['pcl'] = time.time() - t0
    #    pointcloud_out_file = os.path.join(
    #        pointcloud_dir, '%s.ply' % modelname)
    #    export_pointcloud(pointcloud, pointcloud_out_file)
    #    out_file_dict['pointcloud'] = pointcloud_out_file
   #if cfg['generation']['copy_input']:
        # Save inputs
        #if input_type == 'img':
        inputs_path = os.path.join(in_dir, '%s.jpg' % modelname)
        inputs = data['inputs'].squeeze(0).cpu()
        visualize_data(inputs, 'img', inputs_path)
        out_file_dict['in'] = inputs_path
        #elif input_type == 'voxels':
        #    inputs_path = os.path.join(in_dir, '%s.off' % modelname)
        #    inputs = data['inputs'].squeeze(0).cpu()
        #    voxel_mesh = VoxelGrid(inputs).to_mesh()
        #    voxel_mesh.export(inputs_path)
        #    out_file_dict['in'] = inputs_path
        #elif input_type == 'pointcloud':
        #    inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
        #    inputs = data['inputs'].squeeze(0).cpu().numpy()
        #    export_pointcloud(inputs, inputs_path, False)
        #    out_file_dict['in'] = inputs_path

        # Copy to visualization directory for first vis_n_output samples
        c_it = model_counter[category_id]
        if c_it < cfg['test']['vis_n_outputs']:
            # Save output files
            img_name = '%02d.off' % c_it
            for k, filepath in out_file_dict.items():
                ext = os.path.splitext(filepath)[1]
                out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                    % (c_it, k, ext))
                shutil.copyfile(filepath, out_file)

        model_counter[category_id] += 1
#time_df = pd.DataFrame(time_dicts)
#time_df.set_index(['idx'], inplace=True)
#time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
#time_df_class = time_df.groupby(by=['class name']).mean()
#time_df_class.to_pickle(out_time_file_class)

# Print results
#time_df_class.loc['mean'] = time_df_class.mean()
#print('Timings [s]:')
#print(time_df_class)

    
   
main(cfg) 

