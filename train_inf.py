if __name__ == "__main__":

    from Arguments import parse_train
    args, net_args = parse_train()
    
    from config import *
    args=initialize(args)

    print(f"# Start {args['mode']}")
    print('## Arguments:',args)

    from pathlib import Path
    import json
    from torch.utils.data import DataLoader, random_split

    from data import *
    from model import dMaSIF, Lion
    from data_iteration import train, iterate, CollateData, Compose
    from martinize import ReshapeBB, BB2Martini
    if args['mode']=='train':

        import time
        import torch
        import torch.multiprocessing as mp

        rank_list=[x for x in args['devices'] if x!='cpu']
        args['devices']=rank_list

        if len(rank_list)>1:
            print('## Using devices:' ,', '.join(rank_list))
            print('## Using DDP')
            ddp=True
        else:
            print('## Using device:' ,', '.join(rank_list))  
            ddp=False 

        if args['restart_training'] != "":
            checkpoint = torch.load( os.path.join(os.path.dirname(__file__),"models/",args['restart_training']), map_location=args['devices'][0])
            net=dMaSIF(checkpoint['net_args'])
            net.load_state_dict(checkpoint["model_state_dict"])
            net.to(args['devices'][0])
            optimizer = Lion(net.parameters(), lr=1e-4)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            starting_epoch = checkpoint["epoch"]+1
            best_loss = checkpoint["best_loss"]
        else:
            net = dMaSIF(net_args)
            optimizer = Lion(net.parameters(), lr=1e-4)
            #optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, amsgrad=True)
            starting_epoch = 0
            best_loss = 1e10 
        
        print('# Model loaded')
        print('## Model arguments:',net_args)

        transformations = (
            Compose([CenterPairAtoms(as_single=True), 
                     RandomRotationPairAtoms(as_single=True)])
            if args['random_rotation']
            else Compose([])
        )
    
        pre_transformations=[SurfacePrecompute(net.preprocess_surface, False),
                             GenerateMatchingLabels(args['threshold']),
                             RemoveUnusedKeys(keys=['sequence', 'bb_xyz'])]
        if args['from_bb']:
            pre_transformations=[ReshapeBB(), BB2Martini()]+pre_transformations

        pre_transformations=Compose(pre_transformations)

        print('# Loading datasets')   
        prefix=f'{args["na"].lower()}_'
        if args['no_h']:
            prefix+='no_h_'
        if args['from_bb']:
            prefix+='from_bb_'
        if args['martini']:
            prefix+='martini_'

        full_dataset = NpiDataset(args['data_dir'], args['training_list'],
                    transform=transformations, pre_transform=pre_transformations, 
                    encoders=args['encoders'], prefix=prefix, pre_filter=iface_valid_filter,
                    martini=('12' if args['martini'] and not args['from_bb'] else ''))
        test_dataset = NpiDataset(args['data_dir'], args['testing_list'],
                    transform=transformations, pre_transform=pre_transformations,
                    encoders=args['encoders'], prefix=prefix, pre_filter=iface_valid_filter,
                    martini=('12' if args['martini'] and not args['from_bb'] else ''))

    # Train/Validation split:
        train_nsamples = len(full_dataset)
        val_nsamples = int(train_nsamples * args['validation_fraction'])
        train_nsamples = train_nsamples - val_nsamples
        train_dataset, val_dataset = random_split(
            full_dataset, [train_nsamples, val_nsamples]
        )
        print('## Train nsamples:',train_nsamples)
        print('## Val nsamples:',val_nsamples)
        print('## Test nsamples:',len(test_dataset))

        if not Path("models/").exists():
            Path("models/").mkdir(exist_ok=False)
   
        fulltime=time.time()
        if ddp:
            mp.spawn(train, args=(rank_list, args, (train_dataset,val_dataset,test_dataset), net, optimizer, 
                              starting_epoch, best_loss, args['port']), nprocs=len(rank_list))
        else:
            train(0, rank_list, args, (train_dataset,val_dataset,test_dataset), net, optimizer, 
                              starting_epoch, best_loss, args['port'], ddp=False)
        fulltime=time.time()-fulltime
        print(f'## Execution complete {fulltime} seconds')
    else:

        model_path = os.path.join(os.path.dirname(__file__),"models",args['experiment_name'])
        checkpoint=torch.load(model_path, map_location=args['device'])
        if checkpoint['net_args'].get('encoders')!=None:
            args['encoders']=checkpoint['net_args']['encoders']

        net = dMaSIF(checkpoint['net_args'])
        net = net.to(args['device'])
        net.load_state_dict(checkpoint["model_state_dict"])
            

        print('# Model loaded')
        print('## Model arguments:',checkpoint['net_args'])

        batch_vars = ["xyz_p1", "xyz_p2", "atom_xyz_p1", "atom_xyz_p2"]

        transformations = (
            Compose([CenterPairAtoms(as_single=True), 
                     RandomRotationPairAtoms(as_single=True)])
            if args['random_rotation']
            else Compose([])
        )

        pre_transformations=[SurfacePrecompute(net.preprocess_surface, False),
                                     GenerateMatchingLabels(args['threshold']),
                                     RemoveUnusedKeys(keys=['sequence', 'bb_xyz'])]
        if args['from_bb']:
            if args['na']=='protein':
                pre_transformations=[ReshapeBB(), BB2Martini()]+pre_transformations
            elif args['na'] in ['DNA','RNA','NA']:
                pre_transformations=[ReshapeBB(), 
                                     BB2Martini(chains=['_p1'], molecule='protein'),
                                     BB2Martini(chains=['_p2'], molecule='na')]+pre_transformations
        pre_transformations=Compose(pre_transformations)

        print('# Loading testing set')   
        if args['single_pdb'] != "":
            pdb_l=[args['single_pdb']]
        elif args['pdb_list'] != "":
            with open(args['pdb_list']) as f:
                pdb_l = f.read().splitlines()
        test_dataset=[]
        test_pdb_ids=[]
        for pdb in tqdm(pdb_l):
            pspl=pdb.split(' ')
            pspl[0]=pspl[0].split('.')
            filename=f'{args["data_dir"]}/{pspl[0][0]}.{"pdb" if len(pspl[0])==1 else pspl[0][-1]}'
            if args['protonate']:
                protonate(filename,filename)
            protein_pair=load_protein_pair(filename, args['encoders'], pspl[1], 
                                            pspl[2] if len(pspl)==3 else None, 
                                            martini=('12' if args['martini'] and not args['from_bb'] else ''))
            if protein_pair==None:
                print(f'##! Skipping non-existing files for {pdb}' )
            else:
                protein_pair.idx=pdb
                test_dataset.append(protein_pair)
                test_pdb_ids.append(pdb)
        
        test_dataset = [pre_transformations(data) for data in tqdm(test_dataset)]
        test_dataset = [transformations(data) for data in tqdm(test_dataset)]
        test_dataset = [data for data in test_dataset if data!=None]
            
        print('## Test nsamples:',len(test_dataset))

        test_loader = DataLoader(
            test_dataset, batch_size=args['batch_size'], collate_fn=CollateData(batch_vars), shuffle=False)

        print('# Start prediction')

        if not os.path.isdir(Path(args['out_dir'])):
            os.makedirs(Path(args['out_dir']))

        info = iterate(
                net,
                test_loader,
                None,
                args,
                test=True,
                save_path=args['out_dir']
        )

        json.dump(info, open(args['out_dir']+'/meta.json', 'w'), indent=4)

        print('## Data saved to',args['out_dir'])

        for i, pdbs in enumerate(info['PDB IDs']):
            print('; '.join(pdbs))
            for key in info:
                if key not in ['PDB IDs']:
                    print(f"    {key} {info[key][i]}")


