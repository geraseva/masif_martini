import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from sklearn.metrics import roc_auc_score
from torcheval.metrics.functional import binary_auroc

from tqdm import tqdm
import copy

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
#from lion_pytorch import Lion
import os
import warnings
from data import *
from model import Lion

import gc
from helper import *


def save_protein_batch_single(protein_pair_id, P, save_path, pdb_idx):

    protein_pair_id = protein_pair_id.split(" ")
    pdb_id = protein_pair_id[0] + "_" + protein_pair_id[pdb_idx]

    xyz = P["xyz"]

    inputs = P["input_features"]

    embedding = P["embedding_1"] if pdb_idx == 1 else P["embedding_2"]

    if "preds" in P.keys():
        if P["preds"].shape[1]==1:
            predictions = torch.sigmoid(P["preds"])
        else: 
            predictions = F.softmax(P["preds"], dim=1)
    else:
        predictions=torch.zeros((xyz.shape[0],1))

    if predictions.shape[1]==1:
        labels = P["labels"].unsqueeze(dim=1) if P["labels"] is not None else 0.0 * predictions
    else:
        labels = F.one_hot(P["labels"],predictions.shape[1]) if P["labels"] is not None else 0.0 * predictions    
    
    np.savez(f'{save_path}/{pdb_id}.npz', 
             coords=numpy(xyz),
             inputs=numpy(inputs),
             embeddings=numpy(embedding),
             predictions=numpy(predictions),
             labels=numpy(labels)
            )


def compute_binary_loss(P1, lf=F.binary_cross_entropy_with_logits):

    # binary
    pos_preds = P1["preds"][P1["labels"] > 0]
    neg_preds = P1["preds"][P1["labels"] == 0]

    pos_labels = torch.ones_like(pos_preds)
    neg_labels = torch.zeros_like(neg_preds)

    n_points_sample = len(pos_labels)
    pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
    neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]

    pos_preds = pos_preds[pos_indices]
    pos_labels = pos_labels[pos_indices]
    neg_preds = neg_preds[neg_indices]
    neg_labels = neg_labels[neg_indices]

    preds_concat = torch.cat([pos_preds, neg_preds])
    labels_concat = torch.cat([pos_labels, neg_labels])
    
    loss = lf(preds_concat, labels_concat, reduction='mean')
    
    return loss, preds_concat, labels_concat

def compute_complementary_loss(P1, P2, lf=F.binary_cross_entropy_with_logits):

    # complementary
    pos_descs1 = P1["embedding_1"][P1["edge_labels"],:]
    pos_descs2 = P2["embedding_2"][P2["edge_labels"],:]
    pos_preds = torch.sum(pos_descs1*pos_descs2, axis=-1)

    try:
        pos_descs1_2 = P1["embedding_2"][P1["edge_labels"],:]
        pos_descs2_2 = P2["embedding_1"][P2["edge_labels"],:]
        pos_preds2 = torch.sum(pos_descs1_2*pos_descs2_2, axis=-1)

        pos_preds = torch.cat([pos_preds, pos_preds2], dim=0)
    except KeyError:
        pass

    n_desc_sample = 100
        
    sample_desc1=P1["embedding_1"][P1["labels"] == 1]
    sample_desc2 = torch.randperm(len(P2["embedding_2"]))[:n_desc_sample]
    sample_desc2 = P2["embedding_2"][sample_desc2]
    neg_preds = torch.matmul(sample_desc1, sample_desc2.T).view(-1)

    try:
        sample_desc2_1=P1["embedding_2"][P1["labels"] == 1]
        sample_desc1_2 = torch.randperm(len(P1["embedding_2"]))[:n_desc_sample]
        sample_desc1_2 = P1["embedding_2"][sample_desc1_2]
        neg_preds_2 = torch.matmul(sample_desc2_1, sample_desc1_2.T).view(-1)

        neg_preds = torch.cat([neg_preds, neg_preds_2], dim=0)
    except KeyError:
        pass
    
    pos_labels = torch.ones_like(pos_preds)
    neg_labels = torch.zeros_like(neg_preds)

    n_points_sample = len(pos_labels)
    pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
    neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]

    pos_preds = pos_preds[pos_indices]
    pos_labels = pos_labels[pos_indices]
    neg_preds = neg_preds[neg_indices]
    neg_labels = neg_labels[neg_indices]

    preds_concat = torch.cat([pos_preds, neg_preds])
    labels_concat = torch.cat([pos_labels, neg_labels])
    
    loss = lf(preds_concat, labels_concat, reduction='mean')


    return loss, preds_concat, labels_concat

def extract_single(P_batch, number):
    P = {}  # First and second proteins
    batch = P_batch["batch_xyz"] == number
    batch_atoms = P_batch["batch_atom_xyz"] == number
    if 'batch_sequence' in P_batch:
        batch_seq = P_batch["batch_sequence"] == number

    for key in P_batch.keys():
        if 'atom' in key:
            if ('face' in key) or ('edge' in key):
                P[key] = P_batch.__getitem__(key)
                vert=P[key][:,0] if len(P[key].shape)==2 else P[key]
                P[key] = P[key][batch_atoms[vert]]
                P[key] -= (P_batch["batch_atom_xyz"]<number).sum()
            else:
                P[key] = P_batch.__getitem__(key)[batch_atoms]
        elif ("sequence" in key) or ('bb' in key):
            P[key] = P_batch.__getitem__(key)[batch_seq]
        else:
            if ('face' in key) or ('edge' in key):
                P[key] = P_batch.__getitem__(key)
                vert=P[key][:,0] if len(P[key].shape)==2 else P[key]
                P[key] = P[key][batch[vert]]
                P[key] -= (P_batch["batch_xyz"]<number).sum()
            else:
                P[key] = P_batch.__getitem__(key)[batch]
    return P


def iterate(
    net,
    dataset,
    optimizer,
    args,
    test=False,
    save_path=None
):

    if test:
        net.eval()
        torch.set_grad_enabled(False)
    else:
        net.train()
        torch.set_grad_enabled(True)

    # Statistics and fancy graphs to summarize the epoch:
    info = []
    # Loop over one epoch:
    for protein_pair in tqdm(dataset):  

            #protein_pair.to(args['device'])
       
        if not test:
            optimizer.zero_grad()

        P1_batch = protein_pair.to_dict(chain_idx=1)
        P2_batch = protein_pair.to_dict(chain_idx=2)
   
        outputs = net(P1_batch, P2_batch)
        info_dict=dict(
                       {
                        'surf_time': outputs["surf_time"],
                        "conv_time": outputs["conv_time"],
                        "memory_usage": outputs["memory_usage"],
                       },
                       # Merge the "R_values" dict into "info", with a prefix:
                       **{"R_values/" + k: v for k, v in outputs["R_values"].items()}
                      )

        P1_batch = outputs["P1"]
        P2_batch = outputs["P2"]

        bsampled_preds = None
        bsampled_labels = None
        bsampled_preds2 = None
        bsampled_labels2 = None
        csampled_preds = None
        csampled_labels = None

        if P1_batch["labels"] is not None:
            loss=0
            if net.args['n_outputs']>0:
                bloss, bsampled_preds, bsampled_labels=compute_binary_loss(P1_batch)
                loss+=bloss
                info_dict["binary_loss"]=bloss.detach().item()
            if net.args['asymmetric']:
                bloss2, bsampled_preds2, bsampled_labels2=compute_binary_loss(P2_batch)
                loss+=bloss2
                info_dict["binary_loss2"]=bloss2.detach().item()
            if net.args['split']:
                closs, csampled_preds, csampled_labels=compute_complementary_loss(P1_batch, P2_batch)
                info_dict["complementary_loss"]=closs.detach().item()
                loss+=closs
            info_dict["loss"]=loss.detach().item()

        else:
            pass

        # Compute the gradient, update the model weights:
        if not test:
            loss.backward()
            optimizer.step()   

        if bsampled_labels is not None:
            try: 
                info_dict["binary_AUROC"]=binary_auroc(bsampled_preds.view(-1),bsampled_labels.view(-1)).detach().item()
            except:
                info_dict["binary_AUROC"]=np.NaN
        if bsampled_labels2 is not None:
            try: 
                info_dict["binary_AUROC2"]=binary_auroc(bsampled_preds2.view(-1),bsampled_labels2.view(-1)).detach().item()
            except:
                info_dict["binary_AUROC2"]=np.NaN
        if csampled_labels is not None:
            try: 
                info_dict["complementary_AUROC"]=binary_auroc(csampled_preds.view(-1),csampled_labels.view(-1)).detach().item()
            except:
                info_dict["complementary_AUROC"]=np.NaN

        info.append(info_dict)

        if save_path is not None:
            batch_ids=protein_pair.idx
            if isinstance(batch_ids, str):
                info[-1]['PDB IDs']=[batch_ids]
                save_protein_batch_single(
                            batch_ids, P1_batch, save_path, pdb_idx=1
                        )
                save_protein_batch_single(
                            pdb_id, P2_batch, save_path, pdb_idx=2
                        )
            else:
                info[-1]['PDB IDs']=batch_ids
                for i, pdb_id in enumerate(batch_ids):
                    P1 = extract_single(P1_batch, i)
                    P2 = extract_single(P2_batch, i)

                    save_protein_batch_single(
                            pdb_id, P1, save_path, pdb_idx=1
                    )
                    save_protein_batch_single(
                            pdb_id, P2, save_path, pdb_idx=2
                    )


    # Turn a list of dicts into a dict of lists:
    newdict = {}
    for k, v in [(key, d[key]) for d in info for key in d]:
        if k not in newdict:
            newdict[k] = [v]
        else:
            newdict[k].append(v)

    info = newdict

    gc.collect()
    torch.cuda.empty_cache()
    # Final post-processing:
    return info

def ddp_setup(rank, rank_list, port=12355):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=len(rank_list))
    torch.cuda.set_device(rank_list[rank])

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        args, 
        best_loss = 1e10, 
        ddp=True
    ) -> None:
        self.gpu_id = gpu_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.ddp=ddp
        if ddp:
            self.model = DDP(model, device_ids=[gpu_id])
            self.optimizer = Lion(self.model.parameters(), lr=1e-4)
            self.optimizer.load_state_dict(optimizer.state_dict())
        else:
            self.model = model.to(gpu_id)
            self.optimizer = optimizer

        self.args = args
        self.args['device']=gpu_id
        self.best_loss = best_loss 


    def _run_epoch(self, epoch):

        for dataset_type in ["Train", "Validation", "Test"]:

            if dataset_type == "Train":
                dataloader = self.train_loader
            elif dataset_type == "Validation":
                dataloader = self.val_loader
            elif dataset_type == "Test":
                dataloader = self.test_loader
            if self.ddp:
                dataloader.sampler.set_epoch(epoch)

            # Perform one pass through the data:
            info = iterate(
                self.model,
                dataloader,
                self.optimizer,
                self.args,
                test=(dataset_type!="Train"),
            )
    
            for key, val in info.items():
                if key not in ['PDB IDs']:
                    print(key ,dataset_type , epoch, np.nanmean(val))
    
            if dataset_type == "Validation":  # Store validation loss for saving the model
                val_loss = np.nanmean(info["loss"])
        
                if self.ddp:
                    if val_loss < self.best_loss and self.gpu_id==self.args['devices'][0]:
                        print("## Validation loss {}, saving model".format(val_loss))
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.module.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "best_loss": val_loss,
                                "net_args": self.model.module.args
                            },
                            f"models/{self.args['experiment_name']}"
                        )
                        self.best_loss = val_loss
                else:
                    if val_loss < self.best_loss:
                        print("## Validation loss {}, saving model".format(val_loss))
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "best_loss": val_loss,
                                "net_args": self.model.args
                            },
                            f"models/{self.args['experiment_name']}"
                        )
                        self.best_loss = val_loss


    def train(self, starting_epoch: int):
    
        print('# Start training')
        for i in range(starting_epoch, self.args['n_epochs']):
            torch.cuda.empty_cache()
            self._run_epoch(i)
            

def train(rank: int, rank_list, args, dataset, net, optimizer, starting_epoch, best_loss, port=12355, ddp=True):

    warnings.simplefilter("ignore")
    if ddp:
        ddp_setup(rank, rank_list, port=port)

    batch_vars = ["xyz_p1", "xyz_p2", "atom_xyz_p1", "atom_xyz_p2"]

    if ddp:
        train_loader = DataLoader(
            dataset[0], batch_size=args['batch_size'], collate_fn=CollateData(batch_vars),
            shuffle=False, sampler=DistributedSampler(dataset[0], rank=rank, num_replicas=len(rank_list)))
        val_loader = DataLoader(
            dataset[1], batch_size=args['batch_size'], collate_fn=CollateData(batch_vars),
            shuffle=False, sampler=DistributedSampler(dataset[1], rank=rank, num_replicas=len(rank_list)))
        test_loader = DataLoader(
            dataset[2], batch_size=1, collate_fn=CollateData(batch_vars),
            shuffle=False, sampler=DistributedSampler(dataset[2], rank=rank, num_replicas=len(rank_list)))
    else:
        train_loader = DataLoader(
            dataset[0], batch_size=args['batch_size'], collate_fn=CollateData(batch_vars),
            shuffle=True)
        val_loader = DataLoader(
            dataset[1], batch_size=args['batch_size'], collate_fn=CollateData(batch_vars),
            shuffle=False)
        test_loader = DataLoader(
            dataset[2], batch_size=1, collate_fn=CollateData(batch_vars),
            shuffle=False)

    trainer = Trainer(model=net,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      gpu_id=rank_list[rank],
                      args=args,
                      best_loss=best_loss, ddp=ddp)
    trainer.train(starting_epoch)

    if ddp:
        destroy_process_group()

# these 2 classes are to replace corresponding classes from pyg
class CollateData:

    def __init__(self, follow_batch=['atom_xyz','target_xyz']):
        self.follow_batch=follow_batch

    def __call__(self, data):

        result_dict = {}
        pdb_ids=[]
        for d in data:
            pdb_ids.append(d.idx)
            # add batch
            for key in self.follow_batch:
                if key not in d.keys:
                    continue
                bkey=f'batch_{key}'
                d[bkey]=torch.zeros((d[key].shape[0],), dtype=int).to(d[key].device) 
            # increment
            for key in d.keys:   
                if key not in result_dict:
                    result_dict[key] = d[key]
                else:
                    result_dict[key] = torch.cat((result_dict[key], d[key]+d.__inc__(key, result_dict[key])), dim=0)
                        
        result_dict=PairData(mapping=result_dict)
        result_dict.idx=pdb_ids
        result_dict.contiguous()
        return result_dict

class Compose:
    r"""Composes several transforms together.
    Args:
        transforms (List[Callable]): List of transforms to compose.
    """
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data, skip_errors=True):
        # Shallow-copy the data so that we prevent in-place data modification.
        if not skip_errors:
            return self.forward(copy.copy(data))
        try:
            return self.forward(copy.copy(data))
        except RuntimeError:
            print(f'##! RuntimeError! Failed to transform {data.idx}' )
            return None
        except IndexError:
            print(f'##! IndexError! Failed to transform {data.idx}' )
            return None

    def forward(self,data) :
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
            else:
                data = transform(data)
        return data

