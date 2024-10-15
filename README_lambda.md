# --------------------------------
# STEP 1
# --------------------------------
sudo apt-get update
sudo apt-get install coreutils

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# --------------------------------
# STEP 2
# --------------------------------
## --- OPEN A NEW SHELL ---
conda deactivate

## Clone repo
git clone https://github.com/mattmattkim/benchmarking-gnns.git
cd benchmarking-gnns

# Install python environment
conda env create -f environment_gpu_lamba.yml   
# Activate environment
conda activate benchmark_gnn

# --------------------------------
# STEP 3
# --------------------------------
### Then run
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
CUDA_VISIBLE_DEVICES=0 python main_TSP_edge_classification.py --config configs/TSP_edge_classification_GatedGCN_100k.json --gpu_id 0

