# Reproduce AutoBots on TrajNet++ and Evaluate on Synth-v1

This document provides concise notes on the workflow I have used to perform the
following:
1. prepare the environment,
2. prepare the TrajNet++ and Synth-v1 datasets,
3. reproduce AutoBots results on the synthetic dataset of TrajNet++,
4. train AutoBots on Synth-v1,
5. evaluate AutoBots robustness on Synth-v1.  

## Environment

Prepare the environment (see `README.md` if more details are needed). I
recommend using the correct Python and PyTorch versions (i.e., 3.7 and
1.9.0, respectively). The environment can be prepared, for example, as follows:
```sh
conda create --name autobots python=3.7
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
# pip list | grep torch
# torch                1.9.0+cu111
# torchaudio           0.9.0
# torchvision          0.10.0+cu111
```

## Prepare the datasets

I will first link the respective dataset location in /work/vita, you
might want to create a new folder, e.g., `mkdir data`.
```sh
ln -s /work/vita/datasets/causal_synthetic_data data
```

Download and preprocess the TrajNet++ Data, needed to reproduce the
paper results:
```sh
mkdir data/autobots
mkdir data/autobots/trajnetpp/
wget --directory-prefix=data/autobots/trajnetpp https://github.com/vita-epfl/trajnetplusplusdata/releases/download/v4.0/test.zip
wget --directory-prefix=data/autobots/trajnetpp https://github.com/vita-epfl/trajnetplusplusdata/releases/download/v4.0/train.zip
unzip data/autobots/trajnetpp/test.zip -d data/autobots/trajnetpp
unzip data/autobots/trajnetpp/train.zip -d data/autobots/trajnetpp
tree data/autobots/trajnetpp
#data/autobots/trajnetpp
#├── __MACOSX
#│   ├── test
#│   │   ├── real_data
#│   │   └── synth_data
#│   └── train
#│       └── synth_data
#├── test
#│   ├── real_data
#│   │   ├── biwi_eth.ndjson
#│   │   ├── crowds_uni_examples.ndjson
#│   │   └── crowds_zara02.ndjson
#│   └── synth_data
#│       ├── collision_test.ndjson
#│       └── orca_synth.ndjson
#├── test.zip
#├── train
#│   ├── real_data
#│   │   ├── biwi_hotel.ndjson
#│   │   ├── cff_06.ndjson
#│   │   ├── cff_07.ndjson
#│   │   ├── cff_08.ndjson
#│   │   ├── cff_09.ndjson
#│   │   ├── cff_10.ndjson
#│   │   ├── cff_12.ndjson
#│   │   ├── cff_13.ndjson
#│   │   ├── cff_14.ndjson
#│   │   ├── cff_15.ndjson
#│   │   ├── cff_16.ndjson
#│   │   ├── cff_17.ndjson
#│   │   ├── cff_18.ndjson
#│   │   ├── crowds_students001.ndjson
#│   │   ├── crowds_students003.ndjson
#│   │   ├── crowds_zara01.ndjson
#│   │   ├── crowds_zara03.ndjson
#│   │   ├── lcas.ndjson
#│   │   └── wildtrack.ndjson
#│   └── synth_data
#│       ├── orca_synth.ndjson
#│       └── orca_synth.pkl
#└── train.zip

python datasets/trajnetpp/create_data_npys.py \
  --raw-dataset-path data/autobots/trajnetpp/train/synth_data/ \
  --output-npy-path data/autobots/trajnetpp/
```

Preprocess the Synth-v1 data, assuming it has already been downloaded or
taken from `/work/vita/datasets/causal_synthetic_data`. Note that
Synth-v1 is not publicly available yet. Also note that the
`max-number-of-agents` parameter matches the maximum number of agents in
the standard Synth-v1 dataset (i.e., 12), but not the OOD subsets, that
might have a maximum of 20 or 50 agents.
```sh
mkdir data/autobots/synth-v1/

# Train subset (about 75057 scenes)
python datasets/synth/create_data_npys.py \
  --raw-dataset-path data/synth_v1.a.filtered.train.pkl \
  --output-npy-path data/autobots/synth-v1/train.npy \
  --max-number-of-agents 12

# Val subset (a small of 300 scenes for debugging and fast evaluation, and a larger one of 25019 scenes)
python datasets/synth/create_data_npys.py \
  --raw-dataset-path data/synth_v1.a.filtered.val.300.pkl \
  --output-npy-path data/autobots/synth-v1/val.300.npy \
  --max-number-of-agents 12
python datasets/synth/create_data_npys.py \
  --raw-dataset-path data/synth_v1.a.filtered.val.pkl \
  --output-npy-path data/autobots/synth-v1/val.npy \
  --max-number-of-agents 12

# Test subset (25019 scenes)
python datasets/synth/create_data_npys.py \
  --raw-dataset-path data/synth_v1.a.filtered.test.pkl \
  --output-npy-path data/autobots/synth-v1/test.npy \
  --max-number-of-agents 12

ls -hail data/autobots/synth-v1/
#total 457M
#  1652010 drwxr-xr-x 2 rajic sc-ma3 4.0K Mar  8 02:54 .
#592159162 drwxr-xr-x 4 rajic sc-pme 4.0K Mar  8 02:52 ..
#  1652014 -rw-r--r-- 1 rajic sc-ma3  92M Mar  8 03:56 test.npy
#  1652013 -rw-r--r-- 1 rajic sc-ma3 275M Mar  8 03:56 train.npy
#  1652011 -rw-r--r-- 1 rajic sc-ma3 1.1M Mar  8 03:55 val.300.npy
#  1652012 -rw-r--r-- 1 rajic sc-ma3  92M Mar  8 03:56 val.npy
```

## Training on TrajNet++

Train on the synthetic part of TrajNet++:
```sh
python train.py \
  --exp-id trajnetpp-reproduction-2 \
  --seed 1 \
  --dataset trajnet++ \
  --model-type Autobot-Joint \
  --num-modes 6 \
  --hidden-size 128 \
  --num-encoder-layers 2 \
  --num-decoder-layers 2 \
  --dropout 0.1 \
  --entropy-weight 40.0 \
  --kl-weight 20.0 \
  --use-FDEADE-aux-loss True \
  --tx-hidden-size 384 \
  --batch-size 64 \
  --learning-rate 0.00075 \
  --learning-rate-sched 10 20 30 40 50 \
  --dataset-path data/autobots/trajnetpp
```

Same, but with num-modes 1, for comparison:
```sh
python train.py \
  --exp-id trajnetpp-reproduction \
  --seed 1 \
  --dataset trajnet++ \
  --model-type Autobot-Joint \
  --num-modes 1 \
  --hidden-size 128 \
  --num-encoder-layers 2 \
  --num-decoder-layers 2 \
  --dropout 0.1 \
  --entropy-weight 40.0 \
  --kl-weight 20.0 \
  --use-FDEADE-aux-loss True \
  --tx-hidden-size 384 \
  --batch-size 64 \
  --learning-rate 0.00075 \
  --learning-rate-sched 10 20 30 40 50 \
  --dataset-path data/autobots/trajnetpp
```

## Evaluation on TrajNet++

Evaluate on the synth data in TrajNet++ and verify the reproduction of
the numbers in the AutoBots [paper](https://arxiv.org/abs/2104.00563).
The numbers that need to be reproduced for `num-modes=6` and
`model-type=Autobot-Joint` are `0.095` minADE for evaluation on only the
ego agent and `0.128` minADE for joint evaluation on all agents. These
numbers are reported for the synthetic portion of the TrajNet++
validation subset.

Note that I leave relevant outputs that I've seen on my machine as
comments. Also, note that my training did not last exactly 150 epochs as
it was halted after 120-150 epochs, so you might not see the exact same
numbers I report, but the numbers match the numbers reported in the
paper.

Run the evaluation as follows:
```sh
python evaluate.py --dataset-path data/autobots/trajnetpp \
  --models-path results/trajnet++/Autobot_joint_C6_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_trajnetpp-reproduction_s1/best_models_ade.pth \
  --batch-size 64
#Val dataset loaded with length 5451
#Number of Model Parameters: 2409606
# EGO
#minADE_6: 0.09473745 minADE_10 0.09473745 minADE_5 0.10117042 minFDE_6: 0.17399599 minFDE_1: 0.4844467
# JOINT
#Marg. minADE c: 0.1373357 Marg. minFDE c: 0.30692527
#Scene minADE c: 0.12839976 Scene minFDE c: 0.24307121

python evaluate.py --dataset-path data/autobots/trajnetpp \
  --models-path results/trajnet++/Autobot_joint_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_trajnetpp-reproduction_s1/best_models_ade.pth \
  --batch-size 64
#Val dataset loaded with length 5451
#Number of Model Parameters: 2401286
# EGO
#minADE_1: 0.17721985 minADE_10 0.17721985 minADE_5 0.17721985 minFDE_1: 0.36737996 minFDE_1: 0.36737996
# JOINT
#Marg. minADE c: 0.21922314 Marg. minFDE c: 0.49991363
#Scene minADE c: 0.18508528 Scene minFDE c: 0.37278858
```

## Train and Eval on Synth-v1

Train Autobots-Joint with two hparam configurations. The training lasts
about 4-6 hours and utilizes 70% of one GPU on Izar. You can run the
training and subsequent evaluation as follows:
```sh
python train.py \
  --exp-id synth-A01.002 \
  --seed 1 \
  --dataset synth \
  --model-type Autobot-Joint \
  --num-modes 1 \
  --hidden-size 128 \
  --num-encoder-layers 2 \
  --num-decoder-layers 2 \
  --dropout 0.1 \
  --entropy-weight 40.0 \
  --kl-weight 20.0 \
  --use-FDEADE-aux-loss True \
  --tx-hidden-size 384 \
  --batch-size 128 \
  --learning-rate 0.00075 \
  --learning-rate-sched 10 20 30 40 50 \
  --dataset-path data/autobots/synth-v1/

# Evaluate on the fxull validation subset
python evaluate.py --dataset-path data/autobots/synth-v1/ \
  --models-path results/synth/Autobot_joint_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.002_s1/best_models_ade.pth \
  --batch-size 64 \
  --synth-v1-subset-filename val.npy
#Val dataset loaded with length 25019
#Number of Model Parameters: 2401286
# EGO
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#minADE_1: 0.20999499 minADE_10 0.20999499 minADE_5 0.20999499 minFDE_1: 0.36885577 minFDE_1: 0.36885577
# JOINT
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#Marg. minADE c: 0.16646115 Marg. minFDE c: 0.30197883
#Scene minADE c: 0.1625671 Scene minFDE c: 0.2945742
```

Smaller learning rate:
```sh
python train.py \
  --exp-id synth-A01.003 \
  --seed 1 \
  --dataset synth \
  --model-type Autobot-Joint \
  --num-modes 1 \
  --hidden-size 128 \
  --num-encoder-layers 2 \
  --num-decoder-layers 2 \
  --dropout 0.1 \
  --entropy-weight 40.0 \
  --kl-weight 20.0 \
  --use-FDEADE-aux-loss True \
  --tx-hidden-size 384 \
  --batch-size 64 \
  --learning-rate 0.00001 \
  --learning-rate-sched 10 20 30 40 50 \
  --dataset-path data/autobots/synth-v1/

python evaluate.py --dataset-path data/autobots/synth-v1/ \
  --models-path results/synth/Autobot_joint_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.003_s1/best_models_ade.pth \
  --batch-size 64 \
  --synth-v1-subset-filename val.npy
#Val dataset loaded with length 25019
#Number of Model Parameters: 2401286
# EGO
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#minADE_1: 0.4049575 minADE_10 0.4049575 minADE_5 0.4049575 minFDE_1: 0.6727645 minFDE_1: 0.6727645
# JOINT
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#Marg. minADE c: 0.2987408 Marg. minFDE c: 0.50697494
#Scene minADE c: 0.29615065 Scene minFDE c: 0.5011966
```

Now, train and evaluate Autobot-Ego, a variant that only predicts the
future of the ego agent, not of all agents jointly.

Default learning rate:
```sh
python train.py \
  --exp-id synth-A01.004 \
  --seed 1 \
  --dataset synth \
  --model-type Autobot-Ego \
  --num-modes 1 \
  --hidden-size 128 \
  --num-encoder-layers 2 \
  --num-decoder-layers 2 \
  --dropout 0.1 \
  --entropy-weight 40.0 \
  --kl-weight 20.0 \
  --use-FDEADE-aux-loss True \
  --tx-hidden-size 384 \
  --batch-size 64 \
  --learning-rate 0.00075 \
  --learning-rate-sched 10 20 30 40 50 \
  --dataset-path data/autobots/synth-v1/

python evaluate.py --dataset-path data/autobots/synth-v1/ \
  --models-path results/synth/Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.004_s1/best_models_ade.pth \
  --batch-size 64 \
  --synth-v1-subset-filename val.npy
#Val dataset loaded with length 25019
#Number of Model Parameters: 1226758
# EGO
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#minADE_1: 0.21310575 minADE_10 0.21310575 minADE_5 0.21310575 minFDE_1: 0.37470558 minFDE_1: 0.37470558
```

Lower learning rate:
```sh
python train.py \
  --exp-id synth-A01.005 \
  --seed 1 \
  --dataset synth \
  --model-type Autobot-Ego \
  --num-modes 1 \
  --hidden-size 128 \
  --num-encoder-layers 2 \
  --num-decoder-layers 2 \
  --dropout 0.1 \
  --entropy-weight 40.0 \
  --kl-weight 20.0 \
  --use-FDEADE-aux-loss True \
  --tx-hidden-size 384 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --learning-rate-sched 10 20 30 40 50 \
  --dataset-path data/autobots/synth-v1/

python evaluate.py --dataset-path data/autobots/synth-v1/ \
  --models-path results/synth/Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.005_s1/best_models_ade.pth \
  --batch-size 64 \
  --synth-v1-subset-filename val.npy
#Val dataset loaded with length 25019
#Number of Model Parameters: 1226758
# EGO
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#minADE_1: 0.26375023 minADE_10 0.26375023 minADE_5 0.26375023 minFDE_1: 0.44788545 minFDE_1: 0.44788545
```

Higher learning rate:
```sh
python train.py \
  --exp-id synth-A01.006 \
  --seed 1 \  
  --dataset synth \
  --model-type Autobot-Ego \
  --num-modes 1 \
  --hidden-size 128 \
  --num-encoder-layers 2 \
  --num-decoder-layers 2 \
  --dropout 0.1 \
  --entropy-weight 40.0 \
  --kl-weight 20.0 \
  --use-FDEADE-aux-loss True \
  --tx-hidden-size 384 \
  --batch-size 64 \
  --learning-rate 0.002 \
  --learning-rate-sched 10 20 30 40 50 \
  --dataset-path data/autobots/synth-v1/

python evaluate.py --dataset-path data/autobots/synth-v1/ \
  --models-path results/synth/Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.006_s1/best_models_ade.pth \
  --batch-size 64 \
  --synth-v1-subset-filename val.npy
#Val dataset loaded with length 25019
#Number of Model Parameters: 1226758
# EGO
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#minADE_1: 0.48965982 minADE_10 0.48965982 minADE_5 0.48965982 minFDE_1: 0.7803771 minFDE_1: 0.7803771
```

## Test on Synth-v1

I evaluate the best Ego and the best Joint model on the test set.
```sh
python evaluate.py --dataset-path data/autobots/synth-v1/ \
  --models-path results/synth/Autobot_joint_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.002_s1/best_models_ade.pth \
  --batch-size 64 \
  --synth-v1-subset-filename test.npy
#Val dataset loaded with length 25019
#Number of Model Parameters: 2401286
# EGO
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#minADE_1: 0.21134274 minADE_10 0.21134274 minADE_5 0.21134274 minFDE_1: 0.3700263 minFDE_1: 0.3700263
# JOINT
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#Marg. minADE c: 0.1671212 Marg. minFDE c: 0.3027156
#Scene minADE c: 0.16303936 Scene minFDE c: 0.29489195

python evaluate.py --dataset-path data/autobots/synth-v1/ \
  --models-path results/synth/Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.004_s1/best_models_ade.pth \
  --batch-size 64 \
  --synth-v1-subset-filename test.npy
#Val dataset loaded with length 25019
#Number of Model Parameters: 1226758
# EGO
#0 / 390
#50 / 390
#100 / 390
#150 / 390
#200 / 390
#250 / 390
#300 / 390
#350 / 390
#minADE_1: 0.21464853 minADE_10 0.21464853 minADE_5 0.21464853 minFDE_1: 0.37459657 minFDE_1: 0.37459657
```

## CF Evaluation on Synth-v1

Run the CF evaluation on the Synth-v1 dataset. Results are saved to a
pickle file and then analyzed using utilities from another project
(`m43/explainable-trajectory-prediction`). Note that the produced CSV
file has all the relevant summaries and more.

The CF evaluation is performed sequentially and is therefore slow much
slower than standard evaluation. It takes 30-ish minutes to finish for
the Ego model, and 70-ish minutes for the joint model. For comparison,
the standard, batched evaluation takes less than a minute.

Evaluate Autobots-Joint, and check that you reproduce the numbers reported
in our paper submission:
```sh
python evaluate.py --dataset-path data/autobots/synth-v1/ \
  --models-path results/synth/Autobot_joint_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.002_s1/best_models_ade.pth \
  --batch-size 64 \
  --synth-v1-cf-evaluation \
  --synth-v1-cf-evaluation-raw-synthv1-path data/synth_v1.a.filtered.test.pkl
cd /home/rajic
conda deactivate
source you.sh
#cd /work/vita/frano/you
#module purge
#module load gcc/9.3.0
#conda activate you
#export PYTHONPATH="$PYTHONPATH:$PWD/src"
#export PYTHONPATH="$PYTHONPATH:$PWD/thirdparty/Trajectron_plus_plus/trajectron"
#export PYTHONPATH="$PYTHONPATH:$PWD/thirdparty/PECNet/PECNet/utils"
python -m tools.summarize_results \
  --cf_eval_results_pickle_list "/scratch/izar/rajic/autobots/results/synth/Autobot_joint_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.002_s1/all_future_trajectories__2023.03.08_20.31.25.pkl" \
  --cf_eval_results_name_list "Autobot_joint_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.002_s1__test" \
  --output_csv_path "Autobot_joint_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.002_s1__test.csv" 
```

Evaluate Autobots-Ego, check that you reproduce the numbers reported in
our submission:
```sh
python evaluate.py --dataset-path data/autobots/synth-v1/ \
  --models-path results/synth/Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.004_s1/best_models_ade.pth \
  --batch-size 64 \
  --synth-v1-cf-evaluation \
  --synth-v1-cf-evaluation-raw-synthv1-path data/synth_v1.a.filtered.test.pkl
#cd /work/vita/frano/you
#module purge
#module load gcc/9.3.0
#conda activate you
#export PYTHONPATH="$PYTHONPATH:$PWD/src"
#export PYTHONPATH="$PYTHONPATH:$PWD/thirdparty/Trajectron_plus_plus/trajectron"
#export PYTHONPATH="$PYTHONPATH:$PWD/thirdparty/PECNet/PECNet/utils"
python -m tools.summarize_results \
  --cf_eval_results_pickle_list "/scratch/izar/rajic/autobots/results/synth/Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.004_s1/all_future_trajectories__2023.03.08_19.53.36.pkl" \
  --cf_eval_results_name_list "Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.004_s1__test" \
  --output_csv_path "Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-A01.004_s1__test.csv" 
```