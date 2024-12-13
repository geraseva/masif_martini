
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()

parser.add_argument(
    "--template",
    required=True,
    type=str,
    help="Template structure file",
)

task = parser.add_mutually_exclusive_group(required=True)
task.add_argument("--query",
    type=str,
    help="Query structure file", default=None
    )
task.add_argument("--query_list",
    type=str,
    help="Query structure files list", default=None
    )
parser.add_argument(
    "--query_chains",
    default=None,
    type=str,
    help="Which chains to use in surface computation",
)
parser.add_argument(
    "--query_contigs",
    default=None,
    type=str,
    help="Part of query to focus on",
)
parser.add_argument(
    "--top_hotspot",
    default=0,
    type=int,
    help="How many hotspot residues to return",
)
args=parser.parse_args()

def parse_contigs(contigs):
    out=[]
    for s in contigs.split(','):
        chain=s[0]
        interval=s[1:].split('-')
        if len(interval)==1:
            out.append(f'{chain} {interval[0]}')
        elif len(interval)==2:
            for c in range(int(interval[0]),int(interval[1])+1):
                out.append(f'{chain} {c}')
    return out


from data import protonate, find_modified_residues, load_structure_np, encode_npy
from Bio.PDB import *
import torch
import numpy as np
from model import dMaSIF

import os

model_path = f'{os.path.dirname(__file__)}/models/no_h_prot_no_v'

encoders={'atom_types': [{'name': 'atom_types',
                             'encoder': {'C':0,'O': 1,'N':2,'S':3,'-':4}},
                            {'name': 'atom_rad',
                             'encoder': {'C': 1.70,'N':1.55,'O':1.52,'-':1.80}},
                             {'name': 'mask',
                                'encoder': {"H": 0, "-": 1}}]}

device='cuda' if torch.cuda.is_available() else 'cpu'
print('# Device:',device)

checkpoint=torch.load(model_path, map_location=device)
if checkpoint['net_args'].get('encoders')!=None and len(checkpoint['net_args']['encoders'])>0:
    encoders=checkpoint['net_args']['encoders']
else:
    checkpoint['net_args']['encoders']=encoders

net = dMaSIF(checkpoint['net_args'])
net = net.to(device)
net.load_state_dict(checkpoint["model_state_dict"])
net.eval()
torch.set_grad_enabled(False)

print('# Model loaded')
print('# Model arguments:',checkpoint['net_args'])

print('# Template', args.template)
pdb_file=f'/tmp/{args.template.split("/")[-1]}'
protonate(args.template,pdb_file)
parser = PDBParser(QUIET=True)
structure = parser.get_structure('template', pdb_file)
modified=find_modified_residues(pdb_file)
p=load_structure_np(structure, chain_ids=None, modified=modified)
P1 = encode_npy(p, encoders=encoders)
P1['batch_atom_xyz']=torch.zeros((P1['atom_xyz'].shape[0]), 
                                 device=P1['atom_xyz'].device,
                                 dtype=int)
outputs = net(P1, None)
positive=(outputs['P1']['preds']>0)[:,0]
t_emb=outputs['P1']['embedding_1'][positive].detach()
print(f'# Generated {t_emb.shape[0]} points')

if args.query!=None:
    pdb_list=[args.query]
    if args.query_contigs!=None:
        contig_list=[parse_contigs(args.query_contigs)]
    if args.query_chains!=None:
        chain_names=[args.query_chains.replace(',','').strip()]
    else:
        chain_names=[None]
else:
    pdb_list=[]
    chain_names=[]
    with open(args.query_list,'r') as f:
        for line in f:
            pdb_list.append(line.strip())
    if args.query_contigs!=None:
         if os.path.isfile(args.query_chains):
             contig_list=[]
             with open(args.query_contigs,'r') as f:
                for line in f:
                    contig_list.append(parse_contigs(line.strip()))
    if args.query_chains!=None and os.path.isfile(args.query_chains):
        chain_names=[]
        with open(args.query_chains,'r') as f:
            for line in f:
                chain_names.append(parse_contigs(line.replace(',','').strip()))
    else:
        chain_names=[args.query_chains]*len(pdb_list)

for i,query_pdb in tqdm(enumerate(pdb_list)):
    print('Query', query_pdb)
    pdb_file=f'/tmp/{query_pdb.split("/")[-1]}'
    protonate(query_pdb,pdb_file)
    structure = parser.get_structure('template', pdb_file)
    modified=find_modified_residues(pdb_file)
    p=load_structure_np(structure, chain_ids=chain_names[i], modified=modified)
    P2 = encode_npy(p, encoders=encoders)
    P2['batch_atom_xyz']=torch.zeros((P2['atom_xyz'].shape[0]), 
                                     device=P2['atom_xyz'].device,
                                     dtype=int)    
    outputs = net(P2, None)

    coords=outputs['P1']['xyz'].detach().cpu().numpy()
    q_emb=outputs['P1']['embedding_2'].detach()
    print(f'# Generated {q_emb.shape[0]} points')

    mask=(outputs['P1']['preds']>0)[:,0].detach().cpu().numpy()

    if args.query_contigs!=None:
        contig_atoms=[f"{p['atom_chains'][j]} {res}" in contig_list[i] for j, res in enumerate(p['atom_resids'])]
        dist=((coords[:,None,:]-p['atom_xyz'][None,contig_atoms,:])**2).sum(-1).min(1)
        mask*=(dist<3**2)

    coords=coords[mask]
    q_emb=q_emb[mask]

    prediction=torch.sum(t_emb[None,:,:]*q_emb[:,None,:], dim=-1)
    prediction=torch.sigmoid(prediction).detach().cpu().numpy()

    print('Score', prediction.max(), flush=True)

    if args.top_hotspot>0:
        dist=((coords[:,None,:]-p['atom_xyz'])**2).sum(-1)
        nearest_atoms=dist.argmin(1)
        hotspot_res = {}
        for id in np.unique(nearest_atoms):
            hotspot_res[f"{p['atom_chains'][id]} {p['atom_resids'][id]} {p['atom_resnames'][id]}"]=prediction[nearest_atoms==id].max()

        print('# Hotspot residues')
        for i, w in enumerate(sorted(hotspot_res, key=hotspot_res.get, reverse=True)):
            if i>=args.top_hotspot:
                break
            if hotspot_res[w]<0.5:
                break
            print('#',w, hotspot_res[w])

