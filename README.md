1) Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2) Data (online download in data folder)

./data/
  CIFAR10/ or CIFAR100/ 
  CIFAR-C/
    CIFAR-10-C/
    CIFAR-100-C/

-data_root ./data



If datasets are already present, training will run with:

--data_root ./data

--no_download

3) Train models (this generates the CSVs used by tools)

3.1 SimCLR baseline (seed loop + p loop)

for p in 0.0 0.2 0.3 0.4; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  python -m scripts.train \
    --dataset cifar10 --arch resnet18 --epochs 200 \
    --batch_size 256 --lr 0.3 --tau 0.2 \
    --baseline \
    --pos_corrupt_p $p \
    --lambda_var 0.0 --lambda_cov 0.0 \
    --out_root runs --data_root ./data --no_download
    
done



3.2 CGH-SimCLR (Gate-only) (seed loop + p loop)


for p in 0.0 0.2 0.3 0.4; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  python -m scripts.train \
    --dataset cifar10 --arch resnet18 --epochs 200 \
    --batch_size 256 --lr 0.3 --tau 0.2 \
    --gamma 0.0 --k_gate 0.30 --wmin 0.25 --wmax 4.0 \
    --pos_corrupt_p $p \
    --lambda_var 0.0 --lambda_cov 0.0 \
    --out_root runs --data_root ./data --no_download
    
done


4) Verify outputs exist

After training, each run directory should contain these files:

dynamics.csv

w_hist.csv

robustness_summary.csv

Quick check:

ls runs/**/dynamics.csv
ls runs/**/w_hist.csv
ls runs/**/robustness_summary.csv

5) Training dynamics plots
python tools/plot_training_dynamics.py \
  --inputs "runs/**/dynamics.csv" \
  --outdir plots

6) Weight histogram plots
python tools/plot_weight_histograms.py \
  --inputs "runs/**/w_hist.csv" \
  --outdir plots

7) Merge robustness results (table numbers)
python tools/merge_clean_eval_results.py \
  --inputs "runs/**/robustness_summary.csv" \
  --out_csv plots/robustness_merged.csv

8) Robustness curves (main paper plots)
python tools/plot_robustness_across_datasets.py \
  --inputs "runs/**/robustness_summary.csv" \
  --outdir plots \
  --seeds 0 1 2 3 4 \
  --force_dataset_by_path

9) CIFAR-C evaluation from a checkpoint 

Pick a trained checkpoint:

ls runs/**/ckpt_ep200.pt


Run CIFAR-C eval:

python tools/evaluate_cifar_c.py \
  --ckpt runs/<run_dir>/ckpt_ep200.pt \
  --dataset cifar10 \
  --data_root ./data \
  --c_root ./data/CIFAR-C \
  --out_csv plots/cifarc_eval.csv \
  --no_download


10) Summarize:

python tools/summarize_cifarc_logs.py \
  --in_csv plots/cifarc_eval.csv \
  --out_csv plots/cifarc_summary.csv
