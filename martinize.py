import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os

try:
    from helper import *
except ModuleNotFoundError:
    from .helper import *
    
try:
    with open(os.path.dirname(os.path.abspath(__file__))+'/datasets/ideal_coords.pkl', 'rb') as f:
        ideal_coords = pickle.load(f)
except FileNotFoundError:
    from .prepare_rotamers import ideal_coords, ideal_coords_na

num2aa=['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
num2na=['DA','DC','DG','DT','A','C','G','U']

def nsplit(*x):
    return [i.split() for i in x]
mass = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 32, 'P': 31, 'M': 0}
bb = "N CA C O H H1 H2 H3 O1 O2"
unknown_mapping=nsplit(bb + " CB")
mapping = {
        "ALA":  nsplit(bb + " CB"),
        "CYS":  nsplit(bb, "CB SG"),
        "SEC":  nsplit(bb, "CB SEG"),
        "ASP":  nsplit(bb, "CB CG OD1 OD2"),
        "GLU":  nsplit(bb, "CB CG CD OE1 OE2"),
        "ASX":  nsplit(bb, "CB CG ND1 ND2 OD1 OD2 HD11 HD12 HD21 HD22"),
        "GLX":  nsplit(bb, "CB CG CD OE1 OE2 NE1 NE2 HE11 HE12 HE21 HE22"),
        "PHE":  nsplit(bb, "CB CG CD1 HD1", "CD2 HD2 CE2 HE2", "CE1 HE1 CZ HZ"),
        "GLY":  nsplit(bb),
        "HIS":  nsplit(bb, "CB CG", "CD2 HD2 NE2 HE2", "ND1 HD1 CE1 HE1"),
        "HIH":  nsplit(bb, "CB CG", "CD2 HD2 NE2 HE2", "ND1 HD1 CE1 HE1"),     # Charged Histidine.
        "ILE":  nsplit(bb, "CB CG1 CG2 CD CD1"),
        "LYS":  nsplit(bb, "CB CG CD", "CE NZ HZ1 HZ2 HZ3"),
        "LEU":  nsplit(bb, "CB CG CD1 CD2"),
        "MET":  nsplit(bb, "CB CG SD CE"),
        "ASN":  nsplit(bb, "CB CG ND1 ND2 OD1 OD2 HD11 HD12 HD21 HD22"),
        "PRO":  nsplit(bb, "CB CG CD"),
        "HYP":  nsplit(bb, "CB CG CD OD"),
        "GLN":  nsplit(bb, "CB CG CD OE1 OE2 NE1 NE2 HE11 HE12 HE21 HE22"),
        "ARG":  nsplit(bb, "CB CG CD", "NE HE CZ NH1 NH2 HH11 HH12 HH21 HH22"),
        "SER":  nsplit(bb, "CB OG HG"),
        "THR":  nsplit(bb, "CB OG1 HG1 CG2"),
        "VAL":  nsplit(bb, "CB CG1 CG2"),
        "TRP":  nsplit(bb, "CB CG CD2", "CD1 HD1 NE1 HE1 CE2", "CE3 HE3 CZ3 HZ3", "CZ2 HZ2 CH2 HH2"),
        "TYR":  nsplit(bb, "CB CG CD1 HD1", "CD2 HD2 CE2 HE2", "CE1 HE1 CZ OH HH"),
        "DA": nsplit("N9 C4",     
                     "C8 N7 C5",
                     "P OP1 OP2 O5' H5T O3' H3T O1P O2P",
                     "C5' O4' C4'",
                     "C3' C2' C1'",
                     "C2 N3",
                     "C6 N6 N1"),
        "DG": nsplit("N9 C4",
                     "C8 N7 C5",
                     "P OP1 OP2 O5' H5T O3' H3T O1P O2P",
                     "C5' O4' C4'",
                     "C3' C2' C1'",
                     "C2 N2 N3",
                     "C6 O6 N1"),
        "DC": nsplit("N1 C6",
                     "P OP1 OP2 O5' H5T O3' H3T O1P O2P",
                     "C5' O4' C4'",
                     "C3' C2' C1'",
                     "N3 C2 O2",
                     "C5 C4 N4"),
        "DT": nsplit("N1 C6",
                     "P OP1 OP2 O5' H5T O3' H3T O1P O2P",
                     "C5' O4' C4'",
                     "C3' C2' C1'",   
                     "N3 C2 O2",
                     "C5 C4 O4 C7 C5M"),
        "A":  nsplit("N9 C4",
                     "C8 N7 C5",
                     "P OP1 OP2 O5' H5T O3' H3T O1P O2P",
                     "C5' O4' C4'",
                     "C3' C2' O2' C1'", 
                     "C2 N3",
                     "C6 N6 N1"),
        "G":  nsplit("N9 C4",
                     "C8 N7 C5",
                     "P OP1 OP2 O5' H5T O3' H3T O1P O2P",
                     "C5' O4' C4'",
                     "C3' C2' O2' C1'", 
                     "C2 N2 N3",
                     "C6 O6 N1"),
        "C":  nsplit("N1 C6",
                     "P OP1 OP2 O5' H5T O3' H3T O1P O2P",
                     "C5' O4' C4'",
                     "C3' C2' O2' C1'",
                     "N3 C2 O2",
                     "C5 C4 N4"),
        "U":  nsplit("N1 C6",
                     "P OP1 OP2 O5' H5T O3' H3T O1P O2P",
                     "C5' O4' C4'",
                     "C3' C2' O2' C1'",
                     "N3 C2 O2",
                     "C5 C4 O4 C7 C5M")} 

pseudoatom_types = {
        "ALA":  ['P4'],
        "CYS":  ['P5','C5'],
        "SEC":  ['P5','C5'],
        "ASP":  ['P5','Qa'],
        "GLU":  ['P5','Qa'],
        "ASX":  ['P5','P4'],
        "GLX":  ['P5','P4'], 
        "PHE":  ['P5','SC4','SC4','SC4'],
        "GLY":  ['P5'],
        "HIS":  ['P5','SC4','SP1','SP1'],
        "ILE":  ['P5','AC1'],
        "LYS":  ['P5','C3','P1'],
        "LEU":  ['P5','AC1'],
        "MET":  ['P5','C5'],
        "ASN":  ['P5','P5'],
        "PRO":  ['P5','AC2'],
        "GLN":  ['P5','P4'],
        "ARG":  ['P5','N0','Qd'],
        "SER":  ['P5','P1'],
        "THR":  ['P5','P1'],
        "VAL":  ['P5','AC2'],
        "TRP":  ['P5','SC4','SP1','SC4','SC4'],
        "TYR":  ['P5','SC4','SC4','SP1'],
        "DA": ["TN0", "TNa", "Q0", "SN0","SC2", "TA2", "TA3"],
        "DG": ["TN0", "TNa", "Q0","SN0","SC2", "TG2", "TG3"],
        "DC": ["TN0", "Q0","SN0","SC2", "TY2", "TY3"],
        "DT": ["TN0", "Q0","SN0","SC2", "TT2", "TT3"],
        "A": ["TN0", "TNa", "Q0","SN0","SNda", "TA2", "TA3"],
        "G": ["TN0", "TNa", "Q0","SN0","SNda", "TG2", "TG3"],
        "C": ["TN0", "Q0","SN0","SNda", "TY2", "TY3"],
        "U": ["TN0", "Q0","SN0","SNda", "TT2", "TT3"],
} # from martini_v2.1

pseudoatom_radii = {
    'P5': 4.7, 'AC1': 4.7,'C5': 4.7, 'SP1': 4.7, 
    'N0': 4.7, 'AC2': 4.7,'C3': 4.7, 'P1': 4.7,
    'Qa': 4.7, 'P4': 4.7,'Qd': 4.7, 'SC4': 4.7,
    "Q0": 4.7, "SN0": 4.3, "SC2": 4.3, "SNda": 4.3, 
    "TN0": 3.2, "TA2": 3.2, "TA3": 3.2, "TG2": 3.2, "TG3": 3.2, 
    "TY2": 3.2, "TY3": 3.2, "TT2": 3.2, "TT3": 3.2, "TNa": 3.2    
} # from martini_v2.1

pseudoatom_weights = {
    'P5': 72.0, 'AC1': 72.0, 'C5': 72.0, 'SP1': 45.0, 
    'N0': 72.0, 'AC2': 72.0, 'C3': 72.0, 'P1': 72.0,
    'Qa': 72.0, 'P4': 72.0, 'Qd': 72.0, 'SC4': 45.0, 
    "Q0": 72.0, "SN0": 45.0, "SC2": 45.0, "SNda": 45.0, 
    "TN0": 45.0, "TA2": 45.0, "TA3": 45.0, "TG2": 45.0, "TG3": 45.0, 
    "TY2": 45.0, "TY3": 45.0, "TT2": 45.0, "TT3": 45.0, "TNa": 45.0
} # from martini_v2.1

def martinize(seq, atoms, coords):
    ps_coords=[]
    ps_types=[]
    for i, (aa, ass, xyzs) in enumerate(zip(seq, atoms, coords)):

        av=[]
        for a, xyz in zip(ass, xyzs):
            for j, m in enumerate(mapping.get(aa,unknown_mapping)):
                if len(av)<=j:
                    av.append([[0,0,0],0])
                if a in m:
                    av[j][0][0]+=xyz[0]*mass[a[0]]
                    av[j][0][1]+=xyz[1]*mass[a[0]]
                    av[j][0][2]+=xyz[2]*mass[a[0]]
                    av[j][1]+=mass[a[0]]
        av=[[ps[0][0]/ps[1],ps[0][1]/ps[1],ps[0][2]/ps[1]] for ps in av if ps[1]>0]
        
        ps_coords.append(av)
        ps_types.append(pseudoatom_types[aa][:len(av)])
    return ps_coords, ps_types

init_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
init_CA = torch.zeros_like(init_N)
init_C = torch.tensor([1.5233, 0.000, 0.000]).float()
norm_N = init_N / (torch.norm(init_N, dim=-1, keepdim=True) + 1e-5)
norm_C = init_C / (torch.norm(init_C, dim=-1, keepdim=True) + 1e-5)
cos_ideal_NCAC = torch.sum(norm_N*norm_C, dim=-1) # cosine of ideal N-CA-C bond angle

def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    #N, Ca, C - [B,L, 3]
    #R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    B,L = N.shape[:2]
    
    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix
    
    if non_ideal:
        v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
        cosref = torch.sum(e1*v2, dim=-1) # cosine of current N-CA-C bond angle
        costgt = cos_ideal_NCAC.item()
        cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
        cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
        sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
        Rp = torch.eye(3, device=N.device).repeat(B,L,1,1)
        Rp[:,:,0,0] = cosdel
        Rp[:,:,0,1] = -sindel
        Rp[:,:,1,0] = sindel
        Rp[:,:,1,1] = cosdel
    
        R = torch.einsum('blij,bljk->blik', R,Rp)

    return R, Ca # from https://github.com/baker-laboratory/rf_diffusion_all_atom/blob/main/util.py



class BB2Martini: # to sample pseudoatoms using backbone and aminoacid type data
    def __init__(self, chains=['_p1','_p2'], molecule='protein'):

        self.chains=chains
        
        # get ideal pseudoatom coords and types for aminoacids
        if molecule=='protein':
            ps, ts=martinize(num2aa, 
                     [list(x.keys()) for x in ideal_coords],
                     [list(x.values()) for x in ideal_coords] )
        if molecule=='na':
            ps, ts=martinize(num2na, 
                     [list(x.keys()) for x in ideal_coords_na],
                     [list(x.values()) for x in ideal_coords_na] )
        assert len(ps)==len(ts)

        self.num_aa=len(ts) # A
        self.num_pseudo=max([len(x) for x in ts]) # P

        # encode types
        num2ps=list(set([x for y in ts for x in y ]))
        ps2num={x:i for i, x in enumerate(num2ps)}

        rs=[[pseudoatom_radii[y] for y in x] for x in ts]
        ws=[[pseudoatom_weights[y] for y in x] for x in ts]
        ts=[[ps2num[y] for y in x] for x in ts] 

        self.num_types=len(num2ps) # T
        
        # pad coords and types
        for i in range(len(ps)):
            assert len(ps[i])==len(ts[i])
            to_add=self.num_pseudo-len(ps[i])
            for j in range(to_add):
                ps[i]=[ps[i][0]]+ps[i]
                ts[i]=[ts[i][0]]+ts[i]
                rs[i]=[rs[i][0]]+rs[i]
                ws[i]=[ws[i][0]]+ws[i]
            for j in range(to_add):
                ws[i][j]/=(to_add+1)

        ps=np.array(ps)
        ts=np.array(ts)
        rs=np.array(rs)
        ws=np.array(ws)
        
        self.map_coords=tensor(ps)[None,:,:,:] # (1,A,P,3)
        self.map_types=F.one_hot(inttensor(ts),
                                 num_classes=self.num_types).float()[None,:,:,:] # (1,A,P,T)
        self.map_weights=tensor(ws)[None,:,:,None] # (1,A,P,1)
        self.map_radii=tensor(rs)[None,:,:,None] # (1,A,P,1)
    
    @torch.no_grad()       
    def __call__(self, data):
        
        for ch in self.chains: 
            sequence=data[f'sequence{ch}'][:,:,None,None]
            bb=data[f'bb_xyz{ch}']
                       
            xyz = (sequence*self.map_coords*self.map_weights).sum(1)/(sequence*self.map_weights).sum(1) # (L,P, 3)      
            types = (sequence*self.map_types*self.map_weights).sum(1)/(sequence*self.map_weights).sum(1) # (L, P, T)
            radii = (sequence*self.map_radii*self.map_weights).sum(1)/(sequence*self.map_weights).sum(1) # (L, P, 1)
            weights = (sequence*self.map_weights).sum(1) # (L, P, 1)
        
            Rs, Ts = rigid_from_3_points(bb[None,:,0,:],bb[None,:,1,:],bb[None,:,2,:]) # transforms
            xyz = torch.einsum('lpij,lpj->lpi', Rs, xyz.transpose(0,1)) + Ts
        
            xyz=xyz.transpose(0,1).reshape((-1,3))
            types=types.reshape((-1,self.num_types))
            radii=radii.reshape(-1)
            weights=weights.reshape(-1)
        
            data[f'atom_xyz{ch}']=xyz
            data[f'atom_types{ch}']=types
            data[f'atom_rad{ch}']=radii
            data[f'atom_weights{ch}']=weights

        return data 

class BB2MartiniModule(nn.Module, BB2Martini):
    
    def __init__(self, chains=['_p1','_p2'], molecule='protein'):

        nn.Module.__init__(self)
        BB2Martini.__init__(self,chains, molecule)

        self.map_coords=nn.Parameter(self.map_coords, requires_grad=False)
        self.map_types=nn.Parameter(self.map_types, requires_grad=False)
        self.map_weights=nn.Parameter(self.map_weights, requires_grad=False)
        self.map_radii=nn.Parameter(self.map_radii, requires_grad=False)

    def forward(self, data):

        for ch in self.chains: 
            sequence=data[f'sequence{ch}'][:,:,None,None]
            bb=data[f'bb_xyz{ch}']
                       
            xyz = (sequence*self.map_coords*self.map_weights).sum(1)/(sequence*self.map_weights).sum(1) # (L,P, 3)      
            types = (sequence*self.map_types*self.map_weights).sum(1)/(sequence*self.map_weights).sum(1) # (L, P, T)
            radii = (sequence*self.map_radii*self.map_weights).sum(1)/(sequence*self.map_weights).sum(1) # (L, P, 1)
            weights = (sequence*self.map_weights).sum(1) # (L, P, 1)
        
            Rs, Ts = rigid_from_3_points(bb[None,:,0,:],bb[None,:,1,:],bb[None,:,2,:]) # transforms
            xyz = torch.einsum('lpij,lpj->lpi', Rs, xyz.transpose(0,1)) + Ts
        
            xyz=xyz.transpose(0,1).reshape((-1,3))
            types=types.reshape((-1,self.num_types))
            radii=radii.reshape(-1)
            weights=weights.reshape(-1)

            data[f'atom_xyz{ch}']=xyz
            data[f'atom_types{ch}']=types
            data[f'atom_rad{ch}']=radii
            data[f'atom_weights{ch}']=weights

        return data 


class ReshapeBB: # то сut backbone atoms from the protein

    def __init__(self, chains=['_p1','_p2']):
        self.chains=chains
    
    @torch.no_grad()
    def __call__(self,data):

        for ch in self.chains:                      
            seq=torch.stack([data[f'atom_chain{ch}'],data[f'atom_resid{ch}'], data[f'atom_type{ch}']], dim=1)
            seq, idx=seq.unique(dim=0, sorted=True, return_inverse=True)
            idx=(idx[:,None]==torch.arange(len(seq),device=idx.device)[None,:]).to(int).argmax(dim=0)         
            for key in data.keys:
                if ch not in key:
                    continue
                data[key]=data[key][idx]
            
            seq1=seq[:,:2].unique(dim=0)    
            if len(seq1)*3!=len(data[f'atom_resid{ch}']):
                mask=torch.ones_like(data[f'atom_chain{ch}']).to(bool)
                for aa in seq:
                    idx=(data[f'atom_resid{ch}']==aa[1])*(data[f'atom_chain{ch}']==aa[0])
                    if sum(idx)<3:
                        mask=~idx*mask
                for key in data.keys:
                    if ch not in key:
                        continue
                    data[key]=data[key][mask]   
                
                seq=seq[mask,:2].unique(dim=0)
                assert len(seq)*3==len(data[f'atom_resid{ch}'])
    
            bb_xyz=torch.stack((data[f'atom_xyz{ch}'][0::3],
                                data[f'atom_xyz{ch}'][1::3],
                                data[f'atom_xyz{ch}'][2::3]), 
                               dim=1)
            for key in list(data.keys):
                if ch not in key:
                    continue
                elif 'atom' in key:
                    del data[key]
                else:
                    data[key]=data[key][::3].detach()
            data[f'bb_xyz{ch}']=bb_xyz.detach() # coordinates of 3 backbone atoms

        return data    
    