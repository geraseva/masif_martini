import torch
import torch.nn.functional as F

import os

import pykeops

try:
    pykeops.set_bin_folder(f'.cache/pykeops{pykeops.__version__}/{os.uname().nodename}/')
except AttributeError:
    pykeops.set_build_folder(f'.cache/pykeops{pykeops.__version__}/{os.uname().nodename}/')

from pykeops.torch import LazyTensor

try:
    from martinize import BB2MartiniModule
    from model import dMaSIF
except ModuleNotFoundError:
    from .martinize import BB2MartiniModule #BB2Martini
    from .model import dMaSIF    


num2aa=['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']


def get_O_from_3_points(xyz, bond=1.24, eps=1e-8):
   
    N=xyz[:,0,:]
    CA=xyz[:,1,:]
    C=xyz[:,2,:]

    N=torch.cat((N[1:,:],N[:1,:]), dim=0)

    v=((C-CA)/(torch.norm(C-CA, dim=-1, keepdim=True)+eps) +
       (C-N)/(torch.norm(C-N, dim=-1, keepdim=True)+eps))

    O=C+v/(torch.norm(v, dim=-1, keepdim=True)+eps)*bond

    xyz_out=torch.zeros_like(xyz)
    xyz_out[:,:3,:]=xyz[:,:3,:]
    xyz_out[:,3,:]=O

    return xyz_out

class RFdiff_potential_from_bb:

    def __init__(self, binderlen=-1, int_weight=1, non_int_weight=1, 
                 pos_threshold=3, neg_threshold=5, seq_model_type='protein_mpnn'):

        import LigandMPNN
        from LigandMPNN.model_utils import ProteinMPNN
        from LigandMPNN.data_utils import restype_str_to_int, restype_1to3
        
        restype_3to1={restype_1to3[x]: x for x in restype_1to3.keys()}
        self.renumber_aa=torch.tensor([restype_str_to_int[restype_3to1.get(x,'X')] for x in num2aa], dtype=int)

        path_to_LigandMPNN=LigandMPNN.__path__._path[0]

        self.int_weight=int_weight
        self.non_int_weight=non_int_weight
        self.pos_threshold=pos_threshold
        self.neg_threshold=neg_threshold
        self.binderlen=binderlen

        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize LigandMPNN model
        if seq_model_type == "protein_mpnn":
            checkpoint_path = f"{path_to_LigandMPNN}/model_params/proteinmpnn_v_48_020.pt"
        elif seq_model_type == "ligand_mpnn":
            checkpoint_path = f"{path_to_LigandMPNN}/model_params/ligandmpnn_v_32_010_25.pt"
        elif seq_model_type == "soluble_mpnn":
            checkpoint_path = f"{path_to_LigandMPNN}/model_params/solublempnn_v_48_020.pt"

        seq_checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.seq_model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=seq_checkpoint["num_edges"],
            device=self.device,
            atom_context_num=seq_checkpoint.get('atom_context_num',1),
            model_type=seq_model_type,
            ligand_mpnn_use_side_chain_context=0,
        )
        self.seq_model.load_state_dict(seq_checkpoint["model_state_dict"])
        self.seq_model.to(self.device)
        self.seq_model.eval()
        print('Load LigandMPNN model')

        self.recover_sc=None

        # Initialize dMaSIF
        checkpoint_path=os.path.dirname(os.path.abspath(__file__))+'/models/martini_prot_from_bb'
        surf_checkpoint=torch.load(checkpoint_path, map_location=self.device)
        self.surf_model=dMaSIF(surf_checkpoint['net_args'])
        self.surf_model.load_state_dict(surf_checkpoint["model_state_dict"])
        self.surf_model.to(self.device)
        self.surf_model.eval()
        print('Load dMaSIF model')

    def init_recover_sc(self):
        # Initialize BB2Martini
        
        #self.recover_sc=BB2Martini(chains= ['_p1'] if self.binderlen<0 else ['_p1','_p2'])
        self.recover_sc=BB2MartiniModule(chains= ['_p1'] if self.binderlen<0 else ['_p1','_p2'])
        self.recover_sc.eval()


    def run_LigandMPNN(self, xyz):

        L=xyz.shape[0]

        xyz=get_O_from_3_points(xyz)

        feature_dict = {}
        feature_dict["batch_size"]=1
        feature_dict["S"] = torch.full((1, L),20,dtype=int).to(self.device)# encoded sequence
    
        feature_dict["X"] = xyz[None,:,:4,:] # B*L*4*3 (bb atoms)  ? normalize

        feature_dict["mask"] = torch.ones([1, L]).to(self.device)
        feature_dict["chain_mask"] = torch.ones([1, L]).to(self.device)
        feature_dict["temperature"] = 0.1
        feature_dict["bias"] = torch.zeros((1,L,21)).to(self.device)
        feature_dict["randn"]=torch.randn((1,L)).to(self.device)
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"]=[[]]
        feature_dict["Y"] = torch.zeros([1, L, 16, 3]).to(self.device)
        feature_dict["Y_t"] = torch.zeros([1, L, 16]).to(self.device)
        feature_dict["Y_m"] = torch.zeros([1, L, 16]).to(self.device)

        feature_dict["R_idx"] = torch.arange(L)[None,:].to(self.device) # B*L resnums

        if self.binderlen>0:
            feature_dict["chain_labels"] = torch.cat((torch.zeros((1,self.binderlen)),
                                              torch.ones((1,L-self.binderlen))),1).to(self.device)  # B*L Chain indices
        else:
            feature_dict["chain_labels"]=torch.zeros((1,L)).to(self.device)
    
        output_dict = self.seq_model.score(feature_dict, use_sequence=False)

        return output_dict

    def get_aa_probs(self, xyz):

        output_dict=self.run_LigandMPNN(xyz)
        probs=torch.cat((torch.nn.functional.softmax(output_dict['logits'], dim=-1),
                         torch.zeros_like(output_dict['logits'])),-1) 

        return probs[0,:,self.renumber_aa].contiguous()

    def bb2martini(self, xyz, seq):
     
        feature_dict = {}

        if self.binderlen<0:
            feature_dict['sequence_p1']=seq
            feature_dict['bb_xyz_p1']=xyz[:,:3]
        else:
            feature_dict['sequence_p1']=seq[:self.binderlen]
            feature_dict['sequence_p2']=seq[self.binderlen:]
            feature_dict['bb_xyz_p1']=xyz[:self.binderlen,:3]
            feature_dict['bb_xyz_p2']=xyz[self.binderlen:,:3]

        return self.recover_sc(feature_dict)

    def dmasif(self, d):
        
        for k in list(d.keys()):
            if k in ["xyz_p1", "xyz_p2", "atom_xyz_p1", "atom_xyz_p2", 'sequence_p1','sequence_p2']:
                d[f'batch_{k}']=torch.zeros(d[k].shape[0], dtype=int).to(d[k].device)
     
        P1 = {x[:-3]: d[x] for x in d.keys() if '_p1' in x}
        P2 = None if self.binderlen<0 else {x[:-3]: d[x] for x in d.keys() if '_p2' in x}

        out=self.surf_model(P1, P2)

        return out

    @torch.no_grad()       
    def gen_labels(self, d):
        
        if self.binderlen<0:

            P1=d['P1']
            P1['labels']=torch.zeros_like(P1["preds"])
            P2=None
        else:
            P1=d['P1']
            P2=d['P2']

            xyz1_i = LazyTensor(P1['xyz'][:, None, :])
            xyz2_j = LazyTensor(P2['xyz'][None, :, :])

            xyz_dists = ((xyz1_i - xyz2_j) ** 2).sum(-1)
            xyz_dists = (self.neg_threshold**2 - xyz_dists).step()

            P1['labels'] = (xyz_dists.sum(1) > 1.0).view(-1).detach()
            P2['labels'] = (xyz_dists.sum(0) > 1.0).view(-1).detach()

            pos_xyz1 = P1['xyz'][P1['labels']==1]
            pos_xyz2 = P2['xyz'][P2['labels']==1]

            pos_xyz_dists = (
                ((pos_xyz1[:, None, :] - pos_xyz2[None, :, :]) ** 2).sum(-1)
            )
            edges=torch.nonzero(self.pos_threshold**2 > pos_xyz_dists, as_tuple=True)

            P1['edge_labels']=torch.nonzero(P1['labels'])[edges[0]].view(-1).detach()
            P2['edge_labels']=torch.nonzero(P2['labels'])[edges[1]].view(-1).detach()
   
        return P1, P2

    def calc_loss(self, P1, P2, lf=F.binary_cross_entropy_with_logits):

        binary_loss=0
        complementary_loss=0
        if self.binderlen<0:
            binary_loss+= lf(P1['preds'], P1['labels'], reduction='mean')
        else:
            if P1["edge_labels"].shape[0]>0 and self.int_weight>0:

                pos_descs1 = P1["embedding_1"][P1["edge_labels"],:]
                pos_descs2 = P2["embedding_2"][P2["edge_labels"],:]
                pos_preds = torch.sum(pos_descs1*pos_descs2, axis=-1)

                pos_descs1_2 = P1["embedding_2"][P1["edge_labels"],:]
                pos_descs2_2 = P2["embedding_1"][P2["edge_labels"],:]
                pos_preds2 = torch.sum(pos_descs1_2*pos_descs2_2, axis=-1)

                pos_preds = torch.cat([pos_preds, pos_preds2], dim=0)
                pos_labels = torch.ones_like(pos_preds)

                complementary_loss+=lf(pos_preds, pos_labels, reduction='mean')*self.int_weight
            
            if self.non_int_weight>0:
                       
                if (P1['labels']==0).sum()>0:
                    neg_preds1=P1['preds'][P1['labels']==0]
                    neg_labels1=torch.zeros_like(neg_preds1)
                    binary_loss+=lf(neg_preds1, neg_labels1, reduction='mean')*self.non_int_weight

                if (P2['labels']==0).sum()>0:
                    neg_preds2=P2['preds'][P2['labels']==0]
                    neg_labels2=torch.zeros_like(neg_preds2)
                    binary_loss+=lf(neg_preds2, neg_labels2, reduction='mean')*self.non_int_weight

        return binary_loss, complementary_loss

    def __call__(self, xyz):

        xyz=xyz.clone().to(self.device)

        if xyz.shape[0]<=self.binderlen:
            self.binderlen=-1
        
        if self.recover_sc==None:
            self.init_recover_sc()

        seq=self.get_aa_probs(xyz)

        d=self.bb2martini(xyz, seq)

        d=self.dmasif(d)

        P1, P2=self.gen_labels(d)

        binary_loss, complementary_loss=self.calc_loss(P1, P2)

        print('DMASIF BINARY LOSS:',binary_loss)
        print('DMASIF COMPLEMENTARY LOSS:',complementary_loss)

        return binary_loss+complementary_loss

class ProteinGenerator_potential_from_bb:

    def __init__(self, binderlen=-1, int_weight=1, non_int_weight=1, pos_threshold=3, neg_threshold=5):

        self.int_weight=int_weight
        self.non_int_weight=non_int_weight
        self.pos_threshold=pos_threshold
        self.neg_threshold=neg_threshold
        self.binderlen=binderlen

        self.renumber_aa=torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], dtype=int)

        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        self.recover_sc=None

        # Initialize dMaSIF
        checkpoint_path=os.path.dirname(os.path.abspath(__file__))+'/models/martini_prot_from_bb_no_v'
        surf_checkpoint=torch.load(checkpoint_path, map_location=self.device)
        self.surf_model=dMaSIF(surf_checkpoint['net_args'])
        self.surf_model.load_state_dict(surf_checkpoint["model_state_dict"])
        self.surf_model.to(self.device)
        self.surf_model.eval()
        print('Load dMaSIF model')

    def init_recover_sc(self):
        # Initialize BB2Martini
        
        #self.recover_sc=BB2Martini(chains= ['_p1'] if self.binderlen<0 else ['_p1','_p2'])
        self.recover_sc=BB2MartiniModule(chains= ['_p1'] if self.binderlen<0 else ['_p1','_p2'])
        self.recover_sc.eval()

    def bb2martini(self, xyz, seq):
     
        feature_dict = {}

        if self.binderlen<0:
            feature_dict['sequence_p1']=seq
            feature_dict['bb_xyz_p1']=xyz[:,:3]
        else:
            feature_dict['sequence_p1']=seq[:self.binderlen]
            feature_dict['sequence_p2']=seq[self.binderlen:]
            feature_dict['bb_xyz_p1']=xyz[:self.binderlen,:3]
            feature_dict['bb_xyz_p2']=xyz[self.binderlen:,:3]

        return self.recover_sc(feature_dict)

    def dmasif(self, d):
        
        for k in list(d.keys()):
            if k in ["xyz_p1", "xyz_p2", "atom_xyz_p1", "atom_xyz_p2", 'sequence_p1','sequence_p2']:
                d[f'batch_{k}']=torch.zeros(d[k].shape[0], dtype=int).to(d[k].device)
     
        P1 = {x[:-3]: d[x] for x in d.keys() if '_p1' in x}
        P2 = None if self.binderlen<0 else {x[:-3]: d[x] for x in d.keys() if '_p2' in x}

        out=self.surf_model(P1, P2)

        return out

    @torch.no_grad()       
    def gen_labels(self, d):
        
        if self.binderlen<0:

            P1=d['P1']
            P1['labels']=torch.zeros_like(P1["preds"])
            P2=None
        else:
            P1=d['P1']
            P2=d['P2']

            xyz1_i = LazyTensor(P1['xyz'][:, None, :])
            xyz2_j = LazyTensor(P2['xyz'][None, :, :])

            xyz_dists = ((xyz1_i - xyz2_j) ** 2).sum(-1)
            xyz_dists = (self.neg_threshold**2 - xyz_dists).step()

            P1['labels'] = (xyz_dists.sum(1) > 1.0).view(-1).detach()
            P2['labels'] = (xyz_dists.sum(0) > 1.0).view(-1).detach()

            pos_xyz1 = P1['xyz'][P1['labels']==1]
            pos_xyz2 = P2['xyz'][P2['labels']==1]

            pos_xyz_dists = (
                ((pos_xyz1[:, None, :] - pos_xyz2[None, :, :]) ** 2).sum(-1)
            )
            edges=torch.nonzero(self.pos_threshold**2 > pos_xyz_dists, as_tuple=True)

            P1['edge_labels']=torch.nonzero(P1['labels'])[edges[0]].view(-1).detach()
            P2['edge_labels']=torch.nonzero(P2['labels'])[edges[1]].view(-1).detach()
   
        return P1, P2

    def calc_loss(self, P1, P2, lf=F.binary_cross_entropy_with_logits):

        binary_loss=0
        complementary_loss=0
        if self.binderlen<0:
            binary_loss+= lf(P1['preds'], P1['labels'], reduction='mean')
        else:
            if P1["edge_labels"].shape[0]>0 and self.int_weight>0:

                pos_descs1 = P1["embedding_1"][P1["edge_labels"],:]
                pos_descs2 = P2["embedding_2"][P2["edge_labels"],:]
                pos_preds = torch.sum(pos_descs1*pos_descs2, axis=-1)

                pos_descs1_2 = P1["embedding_2"][P1["edge_labels"],:]
                pos_descs2_2 = P2["embedding_1"][P2["edge_labels"],:]
                pos_preds2 = torch.sum(pos_descs1_2*pos_descs2_2, axis=-1)

                pos_preds = torch.cat([pos_preds, pos_preds2], dim=0)
                pos_labels = torch.ones_like(pos_preds)

                complementary_loss+=lf(pos_preds, pos_labels, reduction='mean')*self.int_weight
            
            if self.non_int_weight>0:
                       
                if (P1['labels']==0).sum()>0:
                    neg_preds1=P1['preds'][P1['labels']==0]
                    neg_labels1=torch.zeros_like(neg_preds1)
                    binary_loss+=lf(neg_preds1, neg_labels1, reduction='mean')*self.non_int_weight

                if (P2['labels']==0).sum()>0:
                    neg_preds2=P2['preds'][P2['labels']==0]
                    neg_labels2=torch.zeros_like(neg_preds2)
                    binary_loss+=lf(neg_preds2, neg_labels2, reduction='mean')*self.non_int_weight

        return binary_loss, complementary_loss

    def __call__(self, seq, xyz):

        xyz=xyz.clone().to(self.device)

        if xyz.shape[0]<=self.binderlen:
            self.binderlen=-1
        
        if self.recover_sc==None:
            self.init_recover_sc()
        
        seq=seq[:,self.renumber_aa]

        d=self.bb2martini(xyz, seq)

        d=self.dmasif(d)

        P1, P2=self.gen_labels(d)

        binary_loss, complementary_loss=self.calc_loss(P1, P2)

        print('DMASIF BINARY LOSS:',binary_loss, flush=True)
        print('DMASIF COMPLEMENTARY LOSS:',complementary_loss)


        return binary_loss+complementary_loss

if __name__ == "__main__":

    print('Start test')

    get_pot=RFdiff_potential_from_bb(binderlen=50)

    L=100

    xyz=torch.randn((L, 27, 3)).to(torch.cuda.current_device())

    print(get_pot(xyz))
