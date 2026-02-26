import os


def create_sbatch_ap3_hfvr_vw_variant(submit=False):
    for i in range(3, 4):
        fn = f"train_am_dimer_induced{i}.sbatch"
        python_call = f"""
export iter={i}
python3 -u ./train_models.py \\
    --train_apnet APNet3-fused-variant \\
    --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \\
    --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \\
    --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \\
    --random_seed $iter \\
    --ap_model_path ./models/ap3_ensemble/$iter/ap3_${{iter}}_variant_cutoff3.pt \\
    --n_epochs 50 \\
    --spec_type_ap 2 \\
    --lr 5e-4 \\
    --ds_in_memory False \\
    --data_dir ./data_dimer_$iter \\
    --n_neuron 256 \\
    --n_embed 10 \\
    --r_cut_im 10 \\
        """
    #--data_dir ${{TMPDIR}}/data_dimer_{i} \\

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP3-{i}-E
#SBATCH -oAP3-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-v100
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
#cp -r ./data_dimer_{i} ${{TMPDIR}}/data_dimer_{i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')


def create_sbatch_ap3_InducedDipoleModel_spice(submit=False):
    for i in range(1, 2):
        fn = f"train_am_dimer_induced{i}.sbatch"
        python_call = f"""
export iter={i}
python train_models.py \
    --data_dir ./data_spice \
    --train_am InducedDipoleModel \
    --am_model_path ./models/spice/idm_atp_am_$iter.pt \
    --n_epochs_atom 50 \
    --use_nn_screening \
    --precompute_hfvr \
    --lr 5e-5 \
    --atom_type_param_model_path ./models/spice/atp_mpnn_1.pt \
    --atom_mpnn_pretrained_path ./models/spice/am_3.pt \
    --ds_use_lmdb \
    --spec_type_am 12
        """
    #--data_dir ${{TMPDIR}}/data_dimer_{i} \\

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JSPICE-IDM-{i}-E
#SBATCH -oSPICE-IDM-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=16 #-G1
#SBATCH -pcpu-small
#SBATCH --mem-per-cpu=28GB
#SBATCH -t96:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

export OMP_NUM_THREADS=16
iter={i}
export scratch_dir=${{TMPDIR}}
mkdir -p ${{scratch_dir}}/processed/
mkdir -p ${{scratch_dir}}/raw/
touch ${{scratch_dir}}/raw/1600K_train_dimers-fixed.pkl
touch ${{scratch_dir}}/raw/1600K_test_dimers-fixed.pkl
# find ./data_dimer_3/processed/ -name "lmdb_monomer_ap3_spec*" -exec rsync {{}} ${{scratch_dir}}/processed/ \\;
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_ap3_am_spice(submit=False):
    for i in range(1, 2):
        fn = f"train_am_dimer_induced{i}.sbatch"
        python_call = f"""
export iter={i}
python train_models.py \
    --data_dir ./data_spice \
    --train_am AtomModel \
    --am_model_path ./models/spice/am_1.pt \
    --n_epochs_atom 50 \
    --use_nn_screening \
    --precompute_hfvr \
    --lr 5e-5 \
    --atom_type_param_model_path ./models/spice/atp_mpnn_1.pt \
    --atom_mpnn_pretrained_path ./models/spice/am_1.pt \
    --ds_use_lmdb \
    --spec_type_am 12
        """
    #--data_dir ${{TMPDIR}}/data_dimer_{i} \\

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JSPICE-AM-{i}-E
#SBATCH -oSPICE-AM-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=16 #-G1
#SBATCH -pcpu-small
#SBATCH --mem-per-cpu=24GB
#SBATCH -t96:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

export OMP_NUM_THREADS=16
iter={i}
export scratch_dir=${{TMPDIR}}
mkdir -p ${{scratch_dir}}/processed/
mkdir -p ${{scratch_dir}}/raw/
touch ${{scratch_dir}}/raw/1600K_train_dimers-fixed.pkl
touch ${{scratch_dir}}/raw/1600K_test_dimers-fixed.pkl
# find ./data_dimer_3/processed/ -name "lmdb_monomer_ap3_spec*" -exec rsync {{}} ${{scratch_dir}}/processed/ \\;
cp -r ./data_spice/processed/lmdb_atomic_induced_dipole_spec_12
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_ap3_InducedDipoleModel(submit=False):
    for i in range(1, 2):
        fn = f"train_am_dimer_induced{i}.sbatch"
        python_call = f"""
export iter={i}
python train_models.py \
    --data_dir ./data_dimer_$iter \
    --train_am InducedDipoleModel \
    --am_model_path ./models/ap3_ensemble/1/idm_atp_am_$iter.pt \
    --n_epochs_atom 500 \
    --use_nn_screening \
    --precompute_hfvr \
    --lr 5e-5 \
    --atom_type_param_model_path ./models/ap3_ensemble/1/atp_mpnn_1.pt \
    --atom_mpnn_pretrained_path ./models/ap3_ensemble/1/am_3.pt \
    --spec_type_am 10
        """
    #--data_dir ${{TMPDIR}}/data_dimer_{i} \\

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JIDM-{i}-E
#SBATCH -oIDM-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=12 #-G1
#SBATCH -pcpu-small
#SBATCH --mem-per-cpu=6G
#SBATCH -t96:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

export OMP_NUM_THREADS=12
iter={i}
export scratch_dir=${{TMPDIR}}
mkdir -p ${{scratch_dir}}/processed/
mkdir -p ${{scratch_dir}}/raw/
touch ${{scratch_dir}}/raw/1600K_train_dimers-fixed.pkl
touch ${{scratch_dir}}/raw/1600K_test_dimers-fixed.pkl
find ./data_dimer_3/processed/ -name "lmdb_monomer_ap3_spec*" -exec rsync {{}} ${{scratch_dir}}/processed/ \\;
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_ap3_atomInducedDipoleModel(submit=False):
    for i in range(1, 2):
        fn = f"train_am_dimer_induced{i}.sbatch"
#         python_call = f"""
# export iter={i}
# python3 -u ./train_models.py \
#     --train_am AtomInducedDipoleModel \
#     --am_model_path ./models/ap3_ensemble/$iter/atomInducedDipole_atp_$iter.pt \
#     --atom_type_param_model_path ./models/ap3_ensemble/1/atp_mpnn_1.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --n_epochs_atom 500 \
#     --n_neuron 64 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_am 10
#         """
        python_call = f"""
export iter={i}
python3 -u ./train_models.py \
    --train_am AtomInducedDipoleModel \
    --am_model_path ./models/ap3_ensemble/$iter/atomInducedDipole_atp_screeningNN_lr_$iter.pt \
    --atom_type_param_model_path ./models/ap3_ensemble/1/atp_mpnn_1.pt \
    --random_seed $iter \
    --lr 5e-5 \
    --n_epochs_atom 500 \
    --n_neuron 64 \
    --data_dir ./data_dimer_$iter \
    --use_nn_screening \
    --precompute_hfvr \
    --spec_type_am 10
        """
    #--data_dir ${{TMPDIR}}/data_dimer_{i} \\

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAIDM-{i}-E
#SBATCH -oAIDM-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=12 #-G1
#SBATCH -pcpu-small
#SBATCH --mem-per-cpu=6G
#SBATCH -t96:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

export OMP_NUM_THREADS=12
iter={i}
export scratch_dir=${{TMPDIR}}
mkdir -p ${{scratch_dir}}/processed/
mkdir -p ${{scratch_dir}}/raw/
touch ${{scratch_dir}}/raw/1600K_train_dimers-fixed.pkl
touch ${{scratch_dir}}/raw/1600K_test_dimers-fixed.pkl
find ./data_dimer_3/processed/ -name "lmdb_monomer_ap3_spec*" -exec rsync {{}} ${{scratch_dir}}/processed/ \\;
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_ap3_fsaptpbe0d4(submit=False):
    for i in range(1, 2):
        fn = f"train_fsapt_{i}.sbatch"
        python_call = f"""
export iter={i}
python3 -u ./train_models.py \
    --train_apnet APNet3-fused \
    --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \
    --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \
    --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \
    --random_seed $iter \
    --ap_model_path ./models/ap3_ensemble/$iter/ap3_fsaptpbe0d4_27k.pt \
    --ap_pretrained_model_path ./models/ap3_ensemble/$iter/ap3_.pt \
    --n_epochs 40 \
    --data_dir ./data_dimer_$iter \
    --spec_type_ap 8 \
    --ds_type fsapt_energies \
    --ds_class_type lmdb \
    --lr 5e-4 \
    --ds_in_memory False \
        """
    #--data_dir ${{TMPDIR}}/data_dimer_{i} \\

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP3-FSDT-{i}-E
#SBATCH -oAP3-FSDT-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-v100
#SBATCH --mem-per-cpu=8G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
#export scratch_dir=${{TMPDIR}}
#mkdir -p ${{scratch_dir}}/processed/
#mkdir -p ${{scratch_dir}}/raw/
#touch ${{scratch_dir}}/raw/1600K_train_dimers-fixed.pkl
#touch ${{scratch_dir}}/raw/1600K_test_dimers-fixed.pkl
# find ./data_dimer_3/processed/ -name "dimer_ap3_fused_*" -exec rsync {{}} ${{scratch_dir}}/processed/ \\;
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_ap3_fsapt(submit=False):
    for i in range(1, 2):
        fn = f"train_am_dimer_induced{i}.sbatch"
        python_call = f"""
export iter={i}
python3 -u ./train_models.py \
    --train_apnet APNet3-fused \
    --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \
    --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \
    --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \
    --random_seed $iter \
    --ap_model_path ./models/ap3_ensemble/$iter/ap3_fsapt.pt \
    --ap_pretrained_model_path ./models/ap3_ensemble/$iter/ap3_.pt \
    --n_epochs 200 \
    --data_dir ./data_dimer_$iter \
    --spec_type_ap 5 \
    --ds_type fsapt_energies \
    --ds_class_type lmdb \
    --lr 5e-3 \
    --ds_in_memory False \
        """
    #--data_dir ${{TMPDIR}}/data_dimer_{i} \\

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP3-{i}-E
#SBATCH -oAP3-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-l40s
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
export scratch_dir=${{TMPDIR}}
mkdir -p ${{scratch_dir}}/processed/
mkdir -p ${{scratch_dir}}/raw/
touch ${{scratch_dir}}/raw/1600K_train_dimers-fixed.pkl
touch ${{scratch_dir}}/raw/1600K_test_dimers-fixed.pkl
# find ./data_dimer_3/processed/ -name "dimer_ap3_fused_*" -exec rsync {{}} ${{scratch_dir}}/processed/ \\;
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_ap3_hfvr_vw(submit=False):
    for i in range(3, 4):
        fn = f"train_am_dimer_induced{i}.sbatch"
        python_call = f"""
export iter={i}
python3 -u ./train_models.py \\
    --train_apnet APNet3-fused \\
    --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \\
    --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \\
    --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \\
    --random_seed $iter \\
    --ap_model_path ./models/ap3_ensemble/$iter/ap3_$iter_hfvr_vw.pt \\
    --n_epochs 50 \\
    --spec_type_ap 2 \\
    --lr 5e-4 \\
    --ds_in_memory False \\
    --data_dir ./data_dimer_$iter \\
        """
    #--data_dir ${{TMPDIR}}/data_dimer_{i} \\

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP3-{i}-E
#SBATCH -oAP3-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-l40s
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
#cp -r ./data_dimer_{i} ${{TMPDIR}}/data_dimer_{i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')


def create_sbatch_ap3_hfvr_vw_scratch(submit=False):
    for i in [0, 3]:
        fn = f"train_am_dimer_induced{i}.sbatch"
        python_call = f"""
export iter={i}
python3 -u ./train_models.py \\
    --train_apnet APNet3-fused \\
    --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \\
    --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \\
    --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \\
    --random_seed $iter \\
    --ap_model_path ./models/ap3_ensemble/$iter/ap3_$iter_hfvr_vw.pt \\
    --n_epochs 50 \\
    --spec_type_ap 2 \\
    --lr 5e-5 \\
    --ds_in_memory False \\
    --data_dir ${{scratch_dir}} \\
        """

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP3-{i}-E
#SBATCH -oAP3-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-l40s
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
export scratch_dir=${{TMPDIR}}
mkdir -p ${{scratch_dir}}/processed/
mkdir -p ${{scratch_dir}}/raw/
touch ${{scratch_dir}}/raw/1600K_train_dimers-fixed.pkl
touch ${{scratch_dir}}/raw/1600K_test_dimers-fixed.pkl
find ./data_dimer_3/processed/ -name "dimer_ap3_fused_*" -exec rsync {{}} ${{scratch_dir}}/processed/ \\;
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_ap3(submit=False):
    for i in range(1, 2):
        fn = f"train_am_dimer_induced{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        python_call = f"""
export iter={i}
# Hirshfeld + Valence widths
# python3 -u ./train_models.py \\
#     --train_apnet AtomTypeParamModel \\
#     --am_model_path ./models/ap3_ensemble/$iter/am_$iter.pt \\
#     --random_seed $iter \\
#     --lr 5e-5 \\
#     --ap_model_path ./models/ap3_ensemble/$iter/am_h+1_$iter.pt \\
#     --n_epochs 250 \\
#     --n_neuron 32 \\
#     --data_dir ./data_dimer_$iter \\
#     --spec_type_ap 10 \\
# 
# # rm ./models/ap3_ensemble/$iter/am_elst_h+1_$iter.pt
# python3 -u ./train_models.py \\
#     --train_apnet AM-DimerParam \\
#     --am_model_path ./models/ap3_ensemble/$iter/am_$iter.pt \\
#     --atom_type_param_model_path ./models/ap3_ensemble/$iter/am_h+1_$iter.pt \\
#     --random_seed $iter \\
#     --ap_model_path ./models/ap3_ensemble/$iter/am_elst_h+1_$iter.pt \\
#     --n_epochs 25 \\
#     --n_neuron 64 \\
#     --n_params 1 \\
#     --data_dir ./data_dimer_$iter \\
#     --spec_type_ap 2 \\
#     --lr 5e-5 \\
#     --dimer_eval_type elst_damping \\
#     --param_start_mean "1.6" \\
#     --param_start_std "0.25" \\
#     --ds_in_memory True

python3 -u ./train_models.py \\
    --train_apnet APNet3-fused \\
    --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \\
    --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \\
    --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \\
    --random_seed $iter \\
    --ap_model_path ./models/ap3_ensemble/$iter/ap3_$iter.pt \\
    --n_epochs 50 \\
    --data_dir ./data_dimer_$iter \\
    --spec_type_ap 2 \\
    --lr 5e-4 \\
    --ds_in_memory False
        """

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP3-{i}-E
#SBATCH -oAP3-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-v100
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')


def create_sbatch_ap3_AMOEBA(submit=False):
    elst_type="CLIFF"
    for i in range(1, 2):
        fn = f"train_am_dimer_induced{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        python_call = f"""
export iter={i}
# Hirshfeld + Valence widths
# python3 -u ./train_models.py \\
#     --train_apnet AtomTypeParamModel \\
#     --am_model_path ./models/ap3_ensemble/$iter/am_$iter.pt \\
#     --random_seed $iter \\
#     --lr 5e-5 \\
#     --ap_model_path ./models/ap3_ensemble/$iter/am_h+1_$iter.pt \\
#     --n_epochs 250 \\
#     --n_neuron 32 \\
#     --data_dir ./data_dimer_$iter \\
#     --spec_type_ap 10 \\
# 
# rm ./models/ap3_ensemble/$iter/am_elst_h+1_$iter.pt
python3 -u ./train_models.py \\
    --train_apnet AM-DimerParam \\
    --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \\
    --atom_type_param_model_path ./models/ap3_ensemble/$iter/am_h+1_3.pt \\
    --random_seed $iter \\
    --ap_model_path ./models/ap3_ensemble/$iter/am_elst_h+1_{elst_type}.pt \\
    --n_epochs 25 \\
    --n_neuron 64 \\
    --n_params 1 \\
    --data_dir ./data_dimer_$iter \\
    --spec_type_ap 2 \\
    --lr 5e-5 \\
    --dimer_eval_type elst_damping \\
    --param_start_mean "1.6" \\
    --param_start_std "0.25" \\
    --elst_damping_type {elst_type} \\
    --ds_in_memory True

# python3 -u ./train_models.py \\
#     --train_apnet APNet3-fused \\
#     --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \\
#     --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \\
#     --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_AMOEBA_3.pt \\
#     --random_seed $iter \\
#     --ap_model_path ./models/ap3_ensemble/$iter/ap3_AMOEBA_$iter.pt \\
#     --n_epochs 50 \\
#     --data_dir ./data_dimer_$iter \\
#     --spec_type_ap 2 \\
#     --elst_damping_type AMOEBA \\
#     --lr 5e-4 \\
#     --ds_in_memory False
        """

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP3-{i}-E
#SBATCH -oAP3-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-v100
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')


def create_sbatch_elst_MPNN(submit=False):
    for i in range(0, 1):
        fn = f"train_am_dimer_induced{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        python_call = f"""
export iter={i}
python3 -u ./train_models.py \\
    --train_apnet AM-DimerParam \\
    --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \\
    --atom_type_param_model_path ./models/ap3_ensemble/$iter/am_h+1_3.pt \\
    --random_seed $iter \\
    --ap_model_path ./models/ap3_ensemble/$iter/am_elst_MPNN_3.pt \\
    --n_epochs 25 \\
    --n_neuron 64 \\
    --n_params 1 \\
    --data_dir ./data_dimer_$iter \\
    --spec_type_ap 2 \\
    --lr 3e-5 \\
    --dimer_eval_type elst_damping \\
    --param_start_mean "2.3" \\
    --param_start_std "0.20" \\
    --DimerProp_model_type "AtomTypeParamMPNN" \\
    --ds_in_memory True
        """

#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP3-{i}-E
#SBATCH -oAP3-{i}-E_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-v100
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_am_dimer_elst_plus_induced(submit=False):
    for i in range(0, 1):
        fn = f"train_am_dimer_induced{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        python_call = """
python3 -u ./train_models.py \\
    --train_apnet AM-DimerParam \\
    --am_model_path ./models/am_ensemble/am_$iter.pt \\
    --atom_type_param_model_path ./models/ap_atomTypeParamModel/am_h+1_$iter.pt \\
    --random_seed $iter \\
    --ap_model_path ./models/ap_atomTypeParamModel_elst_ind_1/am_h+1_$iter.pt \\
    --n_epochs 50 \\
    --n_neuron 64 \\
    --n_params 2 \\
    --data_dir ./data_dimer_$iter \\
    --spec_type_ap 2 \\
    --dimer_eval_type elst_damping__induced_dipole \\
    --lr 5e-5 \\
    --param_start_mean "1.8,0.9" \\
    --param_start_std "0.20,0.55" \\
    --ds_in_memory True

        """
#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAM-{i}-E+ID
#SBATCH -oAM-{i}-E+ID_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-v100
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_am_dimer_induced(submit=False):
    for i in range(3, 4):
        fn = f"train_am_dimer_induced{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        python_call = """

python3 -u ./train_models.py \\
    --train_apnet AM-DimerParam \\
    --am_model_path ./models/am_ensemble/am_1.pt \\
    --atom_type_param_model_path ./models/ap_atomTypeParamModel/am_h+1_1.pt \\
    --ap_model_path ./models/ap_atomTypeParamModel_ind_1/am_h+1_1.pt \\
    --lr 5e-5 \\
    --random_seed $iter \\
    --n_epochs 50 \\
    --n_neuron 64 \\
    --n_params 1 \\
    --data_dir ./data_dimer_$iter \\
    --spec_type_ap 2 \\
    --world_size 1 \\
    --omp_num_threads 8 \\
    --dimer_eval_type induced_dipole \\
    --param_start_mean 0.9 \\
    --param_start_std 0.3 \\
        """
#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAM-{i}-IND
#SBATCH -oAM-{i}-IND_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-l40s
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')


def create_sbatch_am_dimer_induced_HF_multipole(submit=False):
    for i in range(0, 1):
        fn = f"train_am_dimer_induced{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        python_call = """
# python3 -u ./train_models.py \\
#     --train_am "AtomHirshfeldModel" \\
#     --am_model_path ./models/am_ap3_HF_ensemble/am_$iter.pt \\
#     --spec_type_am 10 \\
#     --random_seed $iter \\
#     --n_epochs 380 \\
#     --lr 5e-5 \\
#     --data_dir_atom ./data_dimer_0 \\
#     --data_dir ./data_dimer_0 \\
#     --world_size 1 \\
#     --omp_num_threads 8 \\

python3 -u ./train_models.py \\
    --train_apnet AM-DimerParam \\
    --am_model_path ./models/am_ap3_HF_ensemble/am_$iter.pt \\
    --random_seed $iter \\
    --lr 5e-5 \\
    --ap_model_path ./models/am_ap3_HF_ensemble_dimer/am_dimer_damped_elst_$iter.pt \\
    --n_epochs 30 \\
    --n_neuron 64 \\
    --data_dir ./data_dimer_$iter \\
    --spec_type_ap 2 \\
    --world_size 1 \\
    --omp_num_threads 8 \\
    --dimer_eval_type elst_damping \\
    --param_start_mean 2.0 \\
    --param_start_std 0.1 \\

python3 -u ./train_models.py \\
    --train_apnet AM-DimerParam \\
    --am_model_path ./models/am_ap3_HF_ensemble/am_$iter.pt \\
    --random_seed $iter \\
    --lr 5e-5 \\
    --ap_model_path ./models/am_ap3_HF_ensemble_dimer/am_dimer_induced_dipole_$iter.pt \\
    --n_epochs 30 \\
    --n_neuron 64 \\
    --data_dir ./data_dimer_$iter \\
    --spec_type_ap 2 \\
    --world_size 1 \\
    --omp_num_threads 8 \\
    --dimer_eval_type induced_dipole \\
    --param_start_mean 0.8 \\
    --param_start_std 0.1 \\
        """
#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAM-{i}-HF
#SBATCH -oAM-{i}-HF_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-l40s
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

# t2 uses ap2 original ensemble for training, but on full dataset...
def create_sbatch_am_dimer_elst(submit=False):
    for i in range(3, 4):
        fn = f"train_am_dimer_elst{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
    # --ap_model_path ./models/am_dimer_ensemble/am_dimer_elst_damped_$iter.pt \\
        python_call = """

python3 -u ./train_models.py \\
    --train_apnet AM-DimerParam \\
    --am_model_path ./models/am_ensemble/am_1.pt \\
    --atom_type_param_model_path ./models/ap_atomTypeParamModel/am_h+1_1.pt \\
    --ap_model_path ./models/ap_atomTypeParamModel_ind_1/am_h+1_1.pt \\
    --random_seed $iter \\
    --lr 5e-5 \\
    --ap_model_path ./models/am_dimer_ensemble/am_dimer_elst_damped_$iter.pt \\
    --n_epochs 25 \\
    --n_neuron 64 \\
    --n_params 1 \\
    --dimer_eval_type elst_damping \\
    --data_dir ./data_dimer_$iter \\
    --spec_type_ap 2 \\
    --world_size 1 \\
    --omp_num_threads 8 \\
        """
#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAM-{i}-ELS
#SBATCH -oAM-{i}-ELS_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-l40s
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')


def create_sbatch_am_dimer_elst_pbe0(submit=False):
    for i in range(0, 1):
        fn = f"train_am_dimer_elst{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
    # --ap_model_path ./models/am_dimer_ensemble/am_dimer_elst_damped_$iter.pt \\
        python_call = """

python3 -u ./train_models.py \\
    --train_apnet AM-DimerParam \\
    --am_model_path ./models/am_pbe0_ensemble/am_$iter.pt \\
    --random_seed $iter \\
    --lr 5e-5 \\
    --ap_model_path ./models/am_dimer_pbe0_ensemble/am_dimer_elst_damped_$iter.pt \\
    --n_epochs 25 \\
    --n_neuron 64 \\
    --data_dir ./data_dimer_$iter \\
    --spec_type_ap 2 \\
    --world_size 1 \\
    --omp_num_threads 8 \\
        """
#SBATCH -pcpu-amd
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAM-{i}-P-ELS
#SBATCH -oAM-{i}-P-ELS_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-l40s
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_ap2(submit=False):
    for i in range(0, 5):
        fn = f"train_ap{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        python_call = """python3 -u ./train_models.py \\
    --am_model_path ./models/am_ensemble/am_$iter.pt \\
    --data_dir ./data_$iter \\
    --data_dir_atom ./data_$iter \\
    --random_seed $iter \\
    --train_ap APNet2-fused \\
    --ap_model_path ./models/ap2-fused_ensemble/ap2_$iter.pt \\
    --n_epochs 50 \\
    --spec_type_ap 2 \\
    --lr 5e-4
        """
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP2-t5-{i}
#SBATCH -oAP2-t5-{i}_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH -pgpu-l40s
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
    # --n_epochs 50 \\
    # --lr_decay 0.1 \\
        if submit:
            os.system(f'sbatch {fn}')


def create_sbatch_AM(submit=False):
    for i in range(0, 5):
        fn = f"train_ap{i}.sbatch"
        python_call = """python3 -u ./train_models.py \\
    --am_model_path ./models/qm7x_cmpnn_ensemble/am_$iter.pt \\
    --data_dir ./data_$iter \\
    --data_dir_atom ./data_$iter \\
    --random_seed $iter \\
    --train_am AtomModel \\
    --n_epochs 500 \\
    --spec_type_am 7 \\
    --lr 5e-4
        """
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP2-t5-{i}
#SBATCH -oAP2-t5-{i}_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH -pgpu-l40s
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
        if submit:
            os.system(f'sbatch {fn}')

def create_sbatch_dap2_crystal(submit=False):
    for i in range(0, 1):
        fn = f"train_ap{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JdAP2-{i}
#SBATCH -odAP2-t1-{i}_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH -pgpu-l40s
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}


# python3 -u ./train_models.py \\
#     --am_model_path ./src/apnet_pt/models/am_ensemble/am_$iter.pt \
#     --data_dir ./data_$iter \\
#     --data_dir_atom ./data_$iter \\
#     --random_seed $iter \\
#     --train_ap APNet2 \\
#     --ap_model_path ./models/dapnet2/ap2_0.pt \\
#     --n_epochs 50 \\
#     --spec_type_ap 2 \\
#     --r_cut_im 16.0 \\
#     --lr 0.0005

m1="B3LYP-D3/aug-cc-pVDZ/CP"
m2="CCSD(T)/CBS/CP"
m1_str="B3LYP-D3_aug-cc-pVDZ_CP"
m2_str="CCSD_LP_T_RP_CBS_CP"
output_name="${{m1_str}}_to_${{m2_str}}_${{iter}}.pt"

python3 -u ./train_models.py \
    --train_apnet dAPNet2 \
    --am_model_path ./models/am_ensemble/am_$iter.pt \
    --random_seed $iter \
    --lr 5e-4 \
    --ap_model_path ./models/dapnet2/$output_name \
    --n_epochs 50 \
    --spec_type_ap 2 \
    --r_cut_im 16.0 \\
    --m1 $m1 \
    --m2 $m2 \
""")
        if submit:
            os.system(f'sbatch {fn}')


def create_AM_sbatch(submit=False):
    for i in range(0, 1):
        fn = f"train_am{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAM2-{i}
#SBATCH -oAM2-t1-{i}_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH -pgpu-l40s
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
python3 -u ./train_models.py \\
    --am_model_path ./models/am_pbe0_ensemble/am_$iter.pt \\
    --data_dir_atom ./data_$iter \\
    --random_seed $iter \\
    --n_epochs_atom 500 \\
    --spec_type_am 4 \\
    --train_am AtomModel \\
"
python3 -u ./train_models.py \\
    --am_model_path ./models/am_pbe0_ensemble/am_$iter.pt \\
    --data_dir_atom ./data_$iter \\
    --random_seed $iter \\
    --n_epochs_atom 500 \\
    --spec_type_am 4 \\
    --train_am AtomModel \\
    # --train_am AtomHirshfeldModel \\
""")
    # --lr 2e-3 \\
    # --lr_decay 0.10
        if submit:
            os.system(f'sbatch {fn}')

# SBATCH --open-mode=append
def create_sbatch(submit=False):
    for i in range(0, 1):
        fn = f"train_ap{i}.sbatch"
        # os.system(f"cp -r data_dir_ex data_{i}")
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -JAP3-{i}
#SBATCH -oAP3-t1-{i}_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH -pgpu-l40s
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

iter={i}
echo "
python3 -u ./train_models.py \\
    --am_model_path ./models/am_hf_ensemble/am_$iter.pt \\
    --spec_type_am 1 \\
    --data_dir ./data_$iter \\
    --data_dir_atom ./data_$iter \\
    --random_seed $iter \\
    --lr 5e-4 \\
    --train_ap APNet3 \\
    --ap_model_path ./models/ap3_ensemble/ap3_$iter.pt \\
    --n_epochs 50 \\
    --spec_type_ap 2 \\
    # --n_epochs_atom 350 \\
    # --train_am AtomHirshfeldModel \\
"
python3 -u ./train_models.py \\
    --am_model_path ./models/am_hf_ensemble/am_$iter.pt \\
    --data_dir ./data_$iter \\
    --data_dir_atom ./data_$iter \\
    --random_seed $iter \\
    --train_ap APNet3 \\
    --ap_model_path ./models/ap3_ensemble/ap3_t1_$iter.pt \\
    --n_epochs 50 \\
    --spec_type_ap 2 \\
    --lr 5e-4 \\
    # --lr_decay 0.1 \\
    # --lr 0.002
    # --lr 5e-5 \\
    # --spec_type_am 1 \\
    # --n_epochs_atom 350 \\
    # --train_am AtomHirshfeldModel \\
""")
    # --lr 2e-3 \\
    # --lr_decay 0.10
        if submit:
            os.system(f'sbatch {fn}')


# create_sbatch(True)
# create_sbatch_ap2(True)

# create_sbatch_am_dimer_elst_plus_induced(True)

# TODO: make compatible with ap2 and ap3. switched to ap2 atm...
# create_sbatch_am_dimer_elst_pbe0(True)
# create_sbatch_am_dimer_induced(True)
# create_sbatch_am_dimer_induced_HF_multipole(True)
# create_sbatch_am_dimer_elst(True)
# create_sbatch_dap2_crystal(True)
# create_AM_sbatch(True)
# import torch
# ap2 = torch.load("./models/dapnet2/ap2_0.pt")
# print(ap2)
# create_sbatch_elst_MPNN(True)

# create_sbatch_ap3(True)
# create_sbatch_ap3_AMOEBA(True)


# create_sbatch_ap3_hfvr_vw(True)
# create_sbatch_ap3_hfvr_vw_scratch(True)

# create_sbatch_ap3_fsapt(True)
# create_sbatch_ap3_atomInducedDipoleModel(True)
# create_sbatch_ap3_InducedDipoleModel(True)
# create_sbatch_ap3_InducedDipoleModel_spice(True)
create_sbatch_ap3_am_spice(True)

# create_sbatch_ap3_fsaptpbe0d4(True)
# create_sbatch_ap3_hfvr_vw_variant(True)
