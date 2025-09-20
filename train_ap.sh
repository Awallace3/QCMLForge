export iter=0


# for seed in 0 # 1 2 3 4
# do
#     export iter=$seed
#     python3 -u ./train_models.py \
#         --train_am "AtomHirshfeldModel" \
#         --am_model_path ./models/am_ap3_HF_ensemble/am_$iter.pt \
#         --spec_type_am 10 \
#         --random_seed $iter \
#         --n_epochs 500 \
#         --lr 5e-5 \
#         --data_dir ./data_dimer_0 \
#         --data_dir_atom ./data_dimer_0 \
#         --world_size 1 \
#         --omp_num_threads 16
# done

# for seed in 0 1 2 3 4
# do
#     export iter=$seed
#     python3 -u ./train_models.py \
#         --train_am "AtomHirshfeldModel" \
#         --am_model_path ./models/am_hirshfeld_ensemble/am_$iter.pt \
#         --spec_type_am 1 \
#         --random_seed $iter \
#         --n_epochs 10 \
#         --lr 5e-5 \
#         --data_dir ./data_dir \
#         --world_size 1 \
#         --omp_num_threads 16 
# done

python3 -u ./train_models.py \
    --train_apnet AM-DimerParam \
    --am_model_path ./models/am_ensemble/am_$iter.pt \
    --atom_type_param_model_path ./models/ap_atomTypeParamModel/am_$iter.pt \
    --random_seed $iter \
    --ap_model_path ./models/ap_atomTypeParamModel_elst_ind_1/am_$iter.pt \
    --n_epochs 50 \
    --n_neuron 64 \
    --data_dir ./data_dimer_$iter \
    --spec_type_ap 6 \
    --lr 5e-4 \
    --dimer_eval_type elst_damping__induced_dipole \
    --param_start_mean "1.3,1.0" \
    --param_start_std "0.10,0.10" \

# Hirshfeld + Valence widths
# python3 -u ./train_models.py \
#     --train_apnet AtomTypeParamModel \
#     --am_model_path ./models/am_ensemble/am_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --ap_model_path ./models/ap_atomTypeParamModel/am_$iter.pt \
#     --n_epochs 50 \
#     --n_neuron 64 \
#     --data_dir ./data_dimer_$iter \
#     --spec_type_ap 10 \

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
