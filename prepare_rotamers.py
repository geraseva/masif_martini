import os
import pymol
from pymol import cmd
import numpy as np

import pickle 

def readRotLib():
    # Column indexes in rotamer library..
    ROTLIB="/home/domain/data/prog/pyrosetta/rosetta_database/rotamer/bbdep02.May.sortlib"

    RES  = 0
    PHI  = 1
    PSI  = 2
    PROB = 8
    CHI1 = 9
    CHI2 = 10
    CHI3 = 11
    CHI4 = 12
    
    rotdat = {}
    if os.path.exists(ROTLIB):
        print("File exists: "+ROTLIB)
        with open(ROTLIB, 'r') as f:
            for line in f:
 
                # Parse by whitespace (I believe format is white space and not fixed-width columns)
                dat = line.split()
 
                # Add to rotamer library in memory : 
                #   key format       RES:PHI_BIN:PSI_BIN
                #   value format     PROB, CHI1, CHI2, CHI3, CHI4
                key=dat[RES]+":"+dat[PHI]+":"+dat[PSI]
                if key in rotdat:
                    rotdat[key].append([ dat[PROB], dat[CHI1], dat[CHI2], dat[CHI3], dat[CHI4] ])
                else:
                    rotdat[key] = [ [ dat[PROB], dat[CHI1], dat[CHI2], dat[CHI3], dat[CHI4] ] ]
        return rotdat
 
 
    else:
        print("Couldn't find Rotamer library")

def set_rotamer(sel, chi1, chi2=0,chi3=0,chi4=0):
    at = cmd.get_model("byres ("+sel+")").atom[0]
 
    l = [chi1,chi2,chi3,chi4]
    for i in range(len(CHIS[at.resn])):
        pymol.editing.set_dihedral(sel + ' and name '+CHIS[at.resn][i][0],
                             sel + ' and name '+CHIS[at.resn][i][1],
                             sel + ' and name '+CHIS[at.resn][i][2],
                             sel + ' and name '+CHIS[at.resn][i][3], str(l[i]))
 
    # Remove some objects that got created
    cmd.delete("pk1")
    cmd.delete("pk2")
    cmd.delete("pkmol")

def rotate_3_points(N, Ca, C, non_ideal=False, eps=0):
    
    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(np.linalg.norm(v1)+eps)
    u2 = v2- (e1*v2).sum()*e1
    e2 = u2/(np.linalg.norm(u2)+eps)
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=-1)
    
    return R, Ca

def get_coords_aa(aa):
    cmd.reinitialize()
    cmd.editor.cmd.editor.attach_amino_acid('pk1',aa.lower(),ss=1)
    if aa in CHIS:
        set_rotamer(aa,*rot[aa])
    atomdict={}
    for atom in cmd.get_model("byres ("+aa+")").atom:
        name=atom.name
        if name[0].isdigit():
            name=name[1:]+name[0]
        atomdict[name]=np.array(atom.coord)
    R, Ca=rotate_3_points(atomdict['N'],atomdict['CA'],atomdict['C'])
    for atom in atomdict:
        atomdict[atom]=((atomdict[atom]-Ca)*R.T).sum(1).round(4)    
    
    return atomdict

NA_to_threeNA = { "A" : "atp", "C" : "ctp", "G" : "gtp", "T" : "ttp", "U" : "utp",
                 "DA" : "atp", "DC" : "ctp", "DG" : "gtp", "DT" : "ttp", "DU" : "utp"}

def get_coords_na(na):
    cmd.reinitialize()
    cmd.editor.cmd.editor.attach_nuc_acid('none',NA_to_threeNA[na],('DNA' if 'D' in na else 'RNA'),
                                          object=na, dbl_helix=False)
    atomdict={}
    for atom in cmd.get_model("byres ("+na+")").atom:
        name=atom.name
        if name[0].isdigit():
            name=name[1:]+name[0]
        atomdict[name]=np.array(atom.coord)
    R, Ca=rotate_3_points(atomdict["C1'"],atomdict["C3'"],atomdict["C4'"])
    for atom in atomdict:
        atomdict[atom]=((atomdict[atom]-Ca)*R.T).sum(1).round(4)    
    
    return atomdict
        
# Atoms for each side-chain angle for each residue
CHIS = {}
CHIS["ARG"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ],
                ["CB","CG","CD","NE" ],
                ["CG","CD","NE","CZ" ]
              ]
 
CHIS["ASN"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","OD1" ]
              ]
 
CHIS["ASP"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","OD1" ]
              ]
CHIS["CYS"] = [ ["N","CA","CB","SG" ]
              ]
 
CHIS["GLN"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ],
                ["CB","CG","CD","OE1"]
              ]
 
CHIS["GLU"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ],
                ["CB","CG","CD","OE1"]
              ]
 
CHIS["HIS"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","ND1"]
              ]
 
CHIS["ILE"] = [ ["N","CA","CB","CG1" ],
                ["CA","CB","CG1","CD1" ]
              ]
 
CHIS["LEU"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD1" ]
              ]
 
CHIS["LYS"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ],
                ["CB","CG","CD","CE"],
                ["CG","CD","CE","NZ"]
              ]
 
CHIS["MET"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","SD" ],
                ["CB","CG","SD","CE"]
              ]
 
CHIS["PHE"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD1" ]
              ]
 
CHIS["PRO"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ]
              ]
 
CHIS["SER"] = [ ["N","CA","CB","OG" ]]
 
CHIS["THR"] = [ ["N","CA","CB","OG1" ]
              ]
 
CHIS["TRP"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD1"]
              ]
 
CHIS["TYR"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD1" ]
              ]
 
CHIS["VAL"] = [ ["N","CA","CB","CG1" ]
              ]

num2aa=['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
        'UNK','MAS'] # from https://github.com/baker-laboratory/rf_diffusion_all_atom/blob/main/chemical.py

# load median rotamers for each aminoacid
if not os.path.exists('datasets/ideal_coords.pkl'):
    rotamers=readRotLib()
    rot={}
    for key in rotamers:
        if key.split(':')[0] not in rot:
            rot[key.split(':')[0]]=[]
        for x in rotamers[key]:
            rot[key.split(':')[0]].append([float(y) for y in x])
        #rot[key.split(':')[0]].append([float(x) for x in rotamers[key][0][1:]])

    for key in rot:
        arr=np.array(rot[key])[:,1:]
        w=np.array(rot[key])[:,0]
        arr=arr[:,(arr!=0).any(0)]
        # calculate histogram to find most frequent rotamers
        H, edges=np.histogramdd(arr, bins=36, range=[(-180,180)]*arr.shape[1], weights=w)
        # get rotamers that fall into the most frequent bin
        # and find their median
        mask=((np.digitize(arr, 
                        edges[0],
                        right=True)-1)==np.array(np.unravel_index(H.argmax(),
                                                                    H.shape))[None,:]).all(1)
        rot[key]=np.median(arr[mask],axis=0)

    # calculate ideal atomic coordinates for each aminoacid
    
    ideal_coords=[]
    for aa in num2aa:
        if aa in ['UNK','MAS']:
            aa='ALA'
        ideal_coords.append(get_coords_aa(aa))

    with open('datasets/ideal_coords.pkl','wb') as f:
        pickle.dump(ideal_coords, f)
else:
    with open('datasets/ideal_coords.pkl', 'rb') as f:
        ideal_coords = pickle.load(f)
    
# get_rotamers for nucleic acids

num2na=['DA','DC','DG','DT','A','C','G','U']
if not os.path.exists('datasets/ideal_coords_na.pkl'):
    # calculate atomic coordinates for each nucleotide   
    ideal_coords_na=[]
    for na in num2na:
        if na in ['UNK','MAS']:
            na='U'
        ideal_coords_na.append(get_coords_na(na))

    with open('datasets/ideal_coords_na.pkl','wb') as f:
        pickle.dump(ideal_coords_na, f)
else:
    with open('datasets/ideal_coords_na.pkl', 'rb') as f:
        ideal_coords_na = pickle.load(f)
