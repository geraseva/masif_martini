export PYTHONPATH=${PYTHONPATH}:$(git rev-parse --show-toplevel)

# commands used to download and unpack the dataset of NA-protein interactions:

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt  \
--keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1q8DU3kfeTORHaylOhQ4QVA4wPU1Jv4rQ' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1q8DU3kfeTORHaylOhQ4QVA4wPU1Jv4rQ" -O pdbs.tar.gz && rm -rf /tmp/cookies.txt
mkdir datasets
tar -xzvf pdbs.tar.gz -C datasets/

# prediction of PPI site

python3 train_inf.py train -e full_prot --na protein --n_epochs 50 --port 12356 --devices cuda:2 cuda:3 >> logs/log.txt

python3 train_inf.py train -e no_h_prot --na protein --no_h  --n_epochs 50 --port 12356 --devices cuda:2 cuda:3 >> logs/log.txt

python3 train_inf.py train -e martini_prot --na protein --martini  --n_epochs 50 --port 12356 --devices cuda:2 cuda:3 >> logs/log.txt

python3 train_inf.py train -e from_bb_martini_prot --na protein --from_bb --n_epochs 50 --port 12356 --devices cuda:2 cuda:3 >> logs/log.txt

python3 train_inf.py train -e from_bb_martini_prot_nnthr --na protein --from_bb \
  --knn 12 --knn_threshold 5 --n_epochs 50 --port 12356 --devices cuda:2 cuda:3 >> logs/log.txt