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

I have troubles with "CUDA out of memory" error, which raises after about 8 hours of training. 
When I use parallel computation on multiple gpus, it raises earlier. But the situation is not much better on a single gpu, 
even without use of DDP modules.
```
python3 train_inf.py train -e martini_prot_from_bb --na protein --from_bb --n_epochs 50 --port 12356 --devices cuda:0 cuda:1
```