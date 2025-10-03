# Masif martini

A model based on [dmasif](https://github.com/FreyrS/dMaSIF) framework to analyze macromolecular interactions.
It was trained on heavy atoms, [martini](https://cgmartini.nl/) pseudoatoms, 
or martini pseudoatoms recovered from backbone atoms as most common rotamers. 
The last can be used as an auxiliary potential in [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion). 
More about this project in https://github.com/geraseva/auxiliary_potential

## Installation

Required libraries:
- biopython
- pytorch
- pykeops
- torcheval
- reduce (used only in full-atom models that need protonation)
- pymol (used only for calculation of rotamer matrix which is already calculated (datasets/ideal_coords.pkl))

### Angle-aware rotamers (optional)

Angle-aware pseudoatom placement can use per-(amino acid, phi, psi) ideal coordinates.

- Data: `datasets/ideal_angles_coords.pkl` (optional). If present, it is used when
  `angle_aware=True` in `BB2Martini`/`BB2MartiniModule`. If absent, code falls back to
  standard per-amino-acid coordinates.
- Generation: run `masif_martini/prepare_rotamers.py` to build the file. This step
  requires PyMOL and a bbdep rotamer library (e.g., `bbdep02.May.sortlib`).
- Enabling: in Python, pass `angle_aware=True` (and optionally `angle_bin_size`, default 10°) when constructing:

```
from masif_martini.martinize import BB2MartiniModule
recover_sc = BB2MartiniModule(chains=['_p1'], molecule='protein', angle_aware=True, angle_bin_size=10)
```

Internally, phi/psi angles are computed from backbone N–CA–C positions and binned to match
the rotamer library binning (e.g., -180, -170, ...). For bins with no data, the model
reuses the standard per-amino-acid mapping.


```
# commands used to download and unpack the dataset:

cd masif_martini
gdown https://drive.google.com/uc?id=1q8DU3kfeTORHaylOhQ4QVA4wPU1Jv4rQ
tar -xzvf pdbs.tar.gz -C datasets/

```
## Training and inference

Commands used for training are in the `benchmark.sh` file.
    
Inference can be performed using commands like:
```
python3 train_inf.py inference --device cpu --batch_size 4 \
--experiment_name no_h_prot --na protein \
--pdb_list lists/testing_ppi.txt --out_dir npys/ 

```

## For collaborators

If you have not downloaded the dataset, run this once:
```
cat lists/short_test_ppi.txt | awk '{print "https://files.rcsb.org/view/" $1 ".pdb"}' | parallel --gnu "wget {} -P datasets/raw "
```

Testing angle-aware functional
```
python3 train_inf.py train -e test_run --na protein --from_bb --angle_aware \
--n_epochs 10 --training_list lists/short_test_ppi.txt --testing_list lists/short_test_ppi.txt --no_store
```
