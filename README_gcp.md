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

# Install Nvidia driviers
<!-- 
This didn't work
sudo apt-get update
sudo apt-get install software-properties-common

echo "deb http://security.debian.org/debian-security bullseye-security main" | sudo tee -a /etc/apt/sources.list
sudo apt update
sudo apt install libssl1.1

sudo apt install libcuda1=560.35.03-1 libnvidia-fbc1=560.35.03-1 libnvidia-opticalflow1=560.35.03-1 libnvcuvid1=560.35.03-1
sudo tee -a /etc/apt/sources.list << EOF
deb http://deb.debian.org/debian bullseye main contrib non-free
deb http://deb.debian.org/debian-security/ bullseye-security main contrib non-free
deb http://deb.debian.org/debian bullseye-updates main contrib non-free
EOF

sudo apt update

sudo apt install linux-headers-amd64 nvidia-driver

sudo apt install libcuda1=560.35.03-1 libnvidia-fbc1=560.35.03-1 libnvidia-opticalflow1=560.35.03-1 libnvcuvid1=560.35.03-1

 -->

### But this worked
curl -L https://github.com/GoogleCloudPlatform/compute-gpu-installation/releases/download/cuda-installer-v1.1.0/cuda_installer.pyz --output cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver

### Then run

python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
CUDA_VISIBLE_DEVICES=0 python main_TSP_edge_classification.py --config configs/TSP_edge_classification_GatedGCN_100k.json --gpu_id 0


# Check Nvidia use
sudo nvidia-smi