export PYTHONPATH=${PYTHONPATH}:$(git rev-parse --show-toplevel)

python3 train_inf.py train -e full_prot --na protein --n_epochs 50 --port 12356 --device cuda:0 >> logs/full_log.txt

python3 train_inf.py train -e no_h_prot --na protein --no_h  --n_epochs 50 --port 12356 --device cuda:0 >> logs/no_h_log.txt

python3 train_inf.py train -e martini_prot --na protein --martini  --n_epochs 50 --port 12356 --device cuda:0 >> logs/martini_log.txt

python3 train_inf.py train -e martini_prot_from_bb --na protein --from_bb --n_epochs 50 --port 12356 --device cuda:0 >> logs/from_bb_log.txt

python3 train_inf.py train -e martini_prot_from_bb_nnthr --na protein --from_bb \
  --knn 12 --knn_threshold 5 --n_epochs 50 --port 12356 --device cuda:0 >> logs/log.txt

python3 train_inf.py train -e martini_prot_from_bb_no_v --na protein --from_bb \
  --n_epochs 50 --port 12356 --device cuda:0 --feature_generation AtomNet_MP >> logs/log.txt
