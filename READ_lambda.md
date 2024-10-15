sudo apt-get update
sudo apt-get install coreutils

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# --- OPEN A NEW SHELL ---
conda deactivate

# Install Git
conda install git

# Clone repo
git clone https://github.com/mattmattkim/benchmarking-gnns.git
cd benchmarking-gnns


# Install python environment
conda env create -f environment_gpu.yml   
# Activate environment
conda activate benchmark_gnn

# Install DGL
conda install -c dglteam/label/th24_cu124 dgl
















# Nvidia driver setup
curl -L https://github.com/GoogleCloudPlatform/compute-gpu-installation/releases/download/cuda-installer-v1.1.0/cuda_installer.pyz --output cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver

### Then run

python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
CUDA_VISIBLE_DEVICES=0 python main_TSP_edge_classification.py --config configs/TSP_edge_classification_GatedGCN_100k.json --gpu_id 0

