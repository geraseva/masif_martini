if __name__ == "__main__":

    from Arguments import parse_train
    args, net_args = parse_train()
    
    from config import *
    args=initialize(args)

    print(f"# Start {args['mode']}")
    print('## Arguments:',args)

    from pathlib import Path
    import json

    if args['mode']=='train':

        from data_iteration import load_train_objs, train
        import time
        import torch.multiprocessing as mp
        rank_list=[x for x in args['devices'] if x!='cpu']
        args['devices']=rank_list

        print('## World size:',len(rank_list))
    
        dataset, net, optimizer, starting_epoch, best_loss = load_train_objs(args, net_args)
        if not Path("models/").exists():
            Path("models/").mkdir(exist_ok=False)
   
        fulltime=time.time()
        mp.spawn(train, args=(rank_list, args, dataset, net, optimizer, starting_epoch, best_loss, args['port']), nprocs=len(rank_list))

        fulltime=time.time()-fulltime
        print(f'## Execution complete {fulltime} seconds')
    else:
        from torch.utils.data import DataLoader

        from data import *
        from model import dMaSIF
        from data_iteration import iterate, CollateData, Compose
        from martinize import ReshapeBB, BB2Martini

        model_path = "models/" + args['experiment_name']
        checkpoint=torch.load(model_path, map_location=args['device'])
        if checkpoint['net_args'].get('encoders')!=None:
            args['encoders']=checkpoint['net_args']['encoders']

        net = dMaSIF(checkpoint['net_args'])
        net = net.to(args['device'])
        net.load_state_dict(checkpoint["model_state_dict"])
            

        print('# Model loaded')
        print('## Model arguments:',checkpoint['net_args'])

        batch_vars = ["xyz_p1", "xyz_p2", "atom_xyz_p1", "atom_xyz_p2", 'sequence_p1','sequence_p2']

        transformations = (
            Compose([CenterPairAtoms(as_single=True), 
                     RandomRotationPairAtoms(as_single=True)])
            if args['random_rotation']
            else Compose([])
        )

        pre_transformations=[SurfacePrecompute(net.preprocess_surface, False),
                                     GenerateMatchingLabels(args['threshold'])]
        if args['from_bb']:
            pre_transformations=[ReshapeBB(), BB2Martini()]+pre_transformations
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
            
        print('## Test nsamples:',len(test_dataset))

        test_loader = DataLoader(
            test_dataset, batch_size=args['batch_size'], collate_fn=CollateData(batch_vars), shuffle=False)

        save_predictions_path = Path("preds/" + args['experiment_name'])

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

        for i, pdbs in enumerate(info['PDB IDs']):
            print('; '.join(pdbs))
            for key in info:
                if key not in ['PDB IDs']:
                    print(f"    {key} {info[key][i]}")

        print('## Data saved to',save_predictions_path)

