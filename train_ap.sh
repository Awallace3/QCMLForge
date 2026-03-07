export iter=1


# for seed in 0 # 1 2 3 4
# do
# done

# for seed in 0 # 1 2 3 4
# do
#     export iter=$seed
    # export iter=0
    # python3 -u ./train_models.py \
    #     --train_am "AtomModel" \
    #     --spec_type_am 12 \
    #     --random_seed $iter \
    #     --n_epochs 500 \
    #     --lr 5e-4 \
    #     --data_dir ./data_spice \
    #     --world_size 1 \
    #     --omp_num_threads 16 \
    #     --am_model_path ./models/spice/am_3_feb26.pt \
        # --am_model_path ./models/ap3_ensemble/1/am_3.pt \
# done
export iter=1
#
# AP3-fused FSAPT training
python3 -u ./train_models.py \
    --train_apnet APNet3-fused \
    --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \
    --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \
    --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \
    --random_seed $iter \
    --ap_model_path ./models/ap3_ensemble/$iter/ap3_${iter}_fsapt.pt \
    --ap_pretrained_model_path ./models/ap3_ensemble/$iter/ap3_.pt \
    --n_epochs 50 \
    --data_dir ./data_dimer_1 \
    --spec_type_ap 8 \
    --ds_type fsapt_energies \
    --ds_class_type lmdb \
    --lr 5e-4 \
    --ds_in_memory False \

# /projects/cos-lab-cs207/ds/awallace43/projects/pdb13k_test_set/fsapt_data/main.py

# export iter=1
# Induced Dipole on frozen AtomModel
# python train_models.py \
#     --data_dir ./data_spice \
#     --train_am InducedDipoleModel \
#     --n_epochs_atom 20 \
#     --use_nn_screening \
#     --spec_type_am 12 \
#     --precompute_hfvr \
#     --lr 5e-5 \
#     --am_model_path ./models/spice/idm_atp_am_${iter}_feb26.pt \
#     --atom_type_param_model_path ./models/ap3_ensemble/1/atp_mpnn_1_feb26.pt \
#     --atom_mpnn_pretrained_path ./models/spice/am_3_feb26.pt \
    # --data_dir ./data_dimer_$iter \

    # --am_model_path ./models/ap3_ensemble/1/idm_atp_am_$iter.pt \
    # --atom_type_param_model_path ./models/ap3_ensemble/1/atp_mpnn_1.pt \
    # --atom_mpnn_pretrained_path ./models/ap3_ensemble/1/am_3.pt \

# Induced Dipole AtomType
# rm ./tests/test_models/ap3_ensemble_0/atomInducedDipole_atp_screeningNN_lr_$iter.pt
# rm ./tests/test_models/ap3_ensemble_0/atomInducedDipole_atp_screeningNN_q_lr_$iter.pt
# python3 -u ./train_models.py \
#     --train_am AtomInducedDipoleModel \
#     --am_model_path ./tests/test_models/ap3_ensemble_0/atomInducedDipole_atp_screeningNN_q_lr_$iter.pt \
#     --atom_type_param_model_path ./models/ap3_ensemble/1/atp_mpnn_1.pt \
#     --random_seed $iter \
#     --lr 5e-4 \
#     --n_epochs_atom 200 \
#     --n_neuron 64 \
#     --data_dir ./data_dimer_$iter \
#     --use_nn_screening \
#     --precompute_hfvr \
#     --spec_type_am 9
    # --am_model_path ./models/ap3_ensemble/$iter/atomInducedDipole_atp_screeningNN_lr_$iter.pt \
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
# Hirshfeld + Valence widths, AtomTypeParamMPNN
# rm ./models/ap3_ensemble/$iter/atp_mpnn_$iter.pt
# python3 -u ./train_models.py \
#     --train_am AtomTypeParamModel \
#     --am_model_path ./models/ap3_ensemble/$iter/atp_mpnn_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --n_epochs_atom 100 \
#     --n_neuron 32 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_am 10 \
# Hirshfeld + Valence widths, AtomTypeParamNN
# python3 -u ./train_models.py \
#     --train_apnet AtomTypeParamModel \
#     --am_model_path ./models/ap3_ensemble/$iter/am_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --ap_model_path ./models/ap3_ensemble/$iter/am_h+1_$iter.pt \
#     --n_epochs 250 \
#     --n_neuron 32 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_ap 10 \

# Elst Damping AtomType
# python3 -u ./train_models.py \
#     --train_apnet AM-DimerParam \
#     --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \
#     --atom_type_param_model_path ./models/ap3_ensemble/$iter/am_h+1_3.pt \
#     --random_seed $iter \
#     --ap_model_path ./models/ap3_ensemble/$iter/am_elst_h+1_AMOEBA_$iter.pt \
#     --n_epochs 55 \
#     --n_neuron 64 \
#     --n_params 1 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_ap 7 \
#     --lr 5e-5 \
#     --dimer_eval_type elst_damping \
#     --param_start_mean "3.8" \
#     --param_start_std "0.25" \
#     --ds_in_memory True \
#     --elst_damping_type AMOEBA
    # --elst_damping_type CLIFF

# Elst Damping AtomTypeMPNN
# python3 -u ./train_models.py \
#     --train_apnet AM-DimerParam \
#     --am_model_path ./models/ap3_ensemble/$iter/am_$iter.pt \
#     --atom_type_param_model_path ./models/ap3_ensemble/$iter/am_h+1_$iter.pt \
#     --random_seed $iter \
#     --ap_model_path ./models/ap3_ensemble/$iter/am_elst_MPNN_$iter.pt \
#     --n_epochs 55 \
#     --n_neuron 64 \
#     --n_params 1 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_ap 6 \
#     --lr 5e-5 \
#     --dimer_eval_type elst_damping \
#     --param_start_mean "2.2" \
#     --param_start_std "0.50" \
#     --ds_in_memory True \
#     --DimerProp_model_type "AtomTypeParamMPNN"

# APNet3-Fused with Elst Damping AtomType
# rm data_dimer_1/processed/dimer_ap3_fused_*spec_7_*
# export scratch_dir=./scratch
# rm -r ${scratch_dir}
# mkdir -p ${scratch_dir}/processed/
# mkdir -p ${scratch_dir}/raw/
# touch ./${scratch_dir}/raw/1600K_train_dimers-fixed.pkl
# touch ./${scratch_dir}/raw/1600K_test_dimers-fixed.pkl
# cp ./data_dimer_1/raw/t_train_100.pkl ./${scratch_dir}/raw/
# cp ./data_dimer_1/raw/t_test_20.pkl ./${scratch_dir}/raw/
# find ./data_dimer_$iter/processed/ -name "dimer_ap3_fused_*" -exec rsync {} ./${scratch_dir}/processed/ \;
# rm ./ap3_testing_variant.pt
# python3 -u ./train_models.py \
#     --train_apnet APNet3-fused-variant \
#     --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \
#     --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \
#     --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \
#     --random_seed $iter \
#     --ap_model_path ./ap3_testing_variant.pt \
#     --n_epochs 50 \
#     --data_dir data_dimer_1 \
#     --spec_type_ap 7 \
#     --lr 5e-4 \
#     --ds_in_memory False \
#     --ds_class_type lmdb \
#     --n_neuron 256 \
#     --n_rbf 12 \
#     --n_embed 10 \
#     --r_cut_im 10 \

# APNet3-Fused with Elst Damping AtomType (AP2 pretrained)
# python3 -u ./train_models.py \
#     --train_apnet APNet3-fused \
#     --am_model_path ./models/ap3_ensemble/$iter/am_3.pt \
#     --atom_type_param_model_path  ./models/ap3_ensemble/$iter/am_h+1_3.pt \
#     --atom_type_param_model_path2 ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \
#     --random_seed $iter \
#     --ap_model_path ./models/ap3_ensemble/$iter/ap3_${iter}_ap2-pretrained.pt \
#     --n_epochs 55 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_ap 8 \
#     --lr 5e-4 \
#     --ds_in_memory False \
    # --ds_class_type lmdb
    # --ap2_pretrained_model_only ./models/ap2_ensemble/ap2_3.pt \

# Elst + Induced dipole
# rm ./models/ap_atomTypeParamModel_elst_ind_1/am_h+1_$iter.pt
# python3 -u ./train_models.py \
#     --train_apnet AM-DimerParam \
#     --am_model_path ./models/am_ensemble/am_$iter.pt \
#     --atom_type_param_model_path ./models/ap_atomTypeParamModel/am_h+1_$iter.pt \
#     --random_seed $iter \
#     --ap_model_path ./models/ap_atomTypeParamModel_elst_ind_1/am_h+1_$iter.pt \
#     --n_epochs 100 \
#     --n_neuron 64 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_ap 5 \
#     --lr 5e-4 \
#     --dimer_eval_type elst_damping__induced_dipole \
#     --param_start_mean "1.8,0.9" \
#     --param_start_std "0.20,0.55" \
#     --ds_in_memory True

# python3 -u ./train_models.py \
#     --train_apnet AM-DimerParam \
#     --am_model_path ./models/am_hirshfeld_ensemble/am_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --ap_model_path ./models/am_dimer_ensemble/am_dimer_induced_dipole_$iter.pt \
#     --n_epochs 50 \
#     --n_neuron 64 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_ap 2 \

# python3 -u ./train_models.py \
#     --train_apnet AM-DimerParam \
#     --am_model_path ./models/am_hirshfeld_ensemble/am_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --ap_model_path ./models/am_dimer_ensemble/am_dimer_induced_dipole_$iter.pt \
#     --n_epochs 50 \
#     --n_neuron 64 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_ap 2 \

# python3 -u ./train_models.py \
#     --train_apnet AM-DimerParam \
#     --am_model_path ./models/am_ensemble/am_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --ap_model_path ./models/am_dimer_ensemble/am_dimer_elst_damp_$iter.pt \
#     --n_epochs 150 \
#     --n_neuron 64 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_ap 2 \

# python3 -u ./train_models.py \
#     --train_apnet APNet3 \
#     --am_model_path ./models/am_hf_ensemble/am_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --ap_model_path ./models/ap3_ensemble/ap3_$iter.pt \
#     --n_epochs 1 \
#     --ds_max_size 100 \

# export iter=0
# python3 -u ./train_models.py \
#     --train_apnet APNet2 \
#     --am_model_path ./models/am_ensemble/am_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --ap_model_path ./models/dapnet2/ap2_$iter.pt \
#     --n_epochs 1 \
#     --r_cut_im 16.0 \
#     --data_dir ./data_dir_dapnet
    # --ds_max_size 100 \

# m1="B3LYP-D3/aug-cc-pVDZ/CP"
# m1="B3LYP-D3/aug-cc-pVTZ/CP"
# m2="CCSD(T)/CBS/CP"
# m1_str="B3LYP-D3_aug-cc-pVDZ_CP"
# m1_str="B3LYP-D3_aug-cc-pVTZ_CP"
# m2_str="CCSD_LP_T_RP_CBS_CP"
# output_name="${m1_str}_to_${m2_str}_${iter}.pt"
# rm -r ./data_dir_dapnet/processed_delta/
#
# # --ds_max_size 100 \
# python3 -u ./train_models.py \
#     --train_apnet dAPNet2 \
#     --am_model_path ./models/am_ensemble/am_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-4 \
#     --ap_model_path ./models/dapnet2/$output_name \
#     --n_epochs 5 \
#     --spec_type_ap 2 \
#     --m1 $m1 \
#     --m2 $m2 \
#     --data_dir ./data_dir_dapnet
