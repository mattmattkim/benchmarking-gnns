mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/.bash_profile
conda activate benchmark_gnn
pip install dill

pip install torch torchvision torchaudio


# Install Debian CUDA drivers for GCP (v11)
wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6


# Regenerate the data using the new libraries


# For preparing the data
python ./data/TSP/generate_TSP.py --min_nodes 50 --max_nodes 500 --num_samples 10000 --filename tsp50-500.txt

# Main
python main_TSP_edge_classification.py --config ./configs/TSP_edge_classification_GatedGCN_100k.json --gpu_id 3