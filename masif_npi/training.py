
# Standard imports:
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path

# Custom data loader and model:
from data import NpiDataset, PairData, CenterPairAtoms
from data import RandomRotationPairAtoms, NormalizeChemFeatures
from model import dMaSIF
from data_iteration import iterate, iterate_surface_precompute
from helper import *
from Arguments import parser
import pickle
import gc

# Parse the arguments, prepare the TensorBoard writer:
args = parser.parse_args()
print('Start training')
print('Arguments:',args)
writer = SummaryWriter("runs/{}".format(args.experiment_name))
model_path = "models/" + args.experiment_name
torch.cuda.set_device(args.device)

if not Path("models/").exists():
    Path("models/").mkdir(exist_ok=False)

# Ensure reproducibility:
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Create the model, with a warm restart if applicable:
net = dMaSIF(args)
net = net.to(args.device)

# We load the train and test datasets.
# Random transforms, to ensure that no network/baseline overfits on pose parameters:
transformations = (
    Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else Compose([NormalizeChemFeatures()])
)

# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
if args.site:
    binary=True
else:
    binary=False
# Load the train dataset:
if not Path('models/dataset.pt').exists():
    full_dataset = NpiDataset(
        "lists/training_npi.list", transform=transformations, binary=binary
    )
    train_loader = DataLoader(
        full_dataset, batch_size=1, follow_batch=batch_vars, shuffle=False
    )
    print("Preprocessing training dataset")
    full_dataset = iterate_surface_precompute(train_loader, net, args)

    with open('models/dataset.pt', 'wb') as fp:
        pickle.dump(full_dataset, fp)
else:
    print('Loading training dataset')
    with open('models/dataset.pt', 'rb') as fp:
        full_dataset=pickle.load(fp)

# Train/Validation split:
train_nsamples = len(full_dataset)
val_nsamples = int(train_nsamples * args.validation_fraction)
train_nsamples = train_nsamples - val_nsamples
train_dataset, val_dataset = random_split(
    full_dataset, [train_nsamples, val_nsamples]
)

# Load the test dataset:
test_dataset = NpiDataset(
    "lists/testing_npi.list", transform=transformations,binary=binary
)
test_loader = DataLoader(
    test_dataset, batch_size=1, follow_batch=batch_vars, shuffle=False
)
print("Preprocessing testing dataset")
test_dataset = iterate_surface_precompute(test_loader, net, args)

# PyTorch_geometric data loaders:
train_loader = DataLoader(
    train_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=1, follow_batch=batch_vars)
test_loader = DataLoader(test_dataset, batch_size=1, follow_batch=batch_vars)


# Baseline optimizer:
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, amsgrad=True)
best_loss = 1e10  # We save the "best model so far"

starting_epoch = 0
if args.restart_training != "":
    checkpoint = torch.load("models/" + args.restart_training, map_location=args.device)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    starting_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]

# Training loop (~100 times) over the dataset:
gc.collect()
for i in range(starting_epoch, args.n_epochs):
    # Train first, Test second:
    for dataset_type in ["Train", "Validation", "Test"]:
        if dataset_type == "Train":
            test = False
        else:
            test = True

        suffix = dataset_type
        if dataset_type == "Train":
            dataloader = train_loader
        elif dataset_type == "Validation":
            dataloader = val_loader
        elif dataset_type == "Test":
            dataloader = test_loader

        # Perform one pass through the data:
        info = iterate(
            net,
            dataloader,
            optimizer,
            args,
            test=test,
            summary_writer=writer,
            epoch_number=i,
        )

        # Write down the results using a TensorBoard writer:


        for key, val in info.items():
            if key in [
                "Loss",
                "ROC-AUC",
                "Distance/Positives",
                "Distance/Negatives",
                "Matching ROC-AUC",
            ]:
                writer.add_scalar(f"{key}/{suffix}", np.mean(val), i)
                print(key ,suffix , i, np.mean(val))
            if "R_values/" in key:
                val = np.array(val)
                writer.add_scalar(f"{key}/{suffix}", np.mean(val[val > 0]), i)

        if dataset_type == "Validation":  # Store validation loss for saving the model
            val_loss = np.mean(info["Loss"])
        

    if True:  # Additional saves
        if val_loss < best_loss:
            print("Validation loss {}, saving model".format(val_loss))
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                model_path + "_epoch{}".format(i),
            )

            best_loss = val_loss
