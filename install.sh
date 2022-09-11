# Run these commands to create your conda environment.
# conda create -n geodb python=3.8
# conda activate geodb

conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install -y habitat-sim withbullet headless -c conda-forge -c aihabitat
pip install -r requirements.txt

# Install the hashgrid encoder.
cd gridencoder
module load cuda/11.1
# Find out what your nvcc path is and use that, for me $which nvcc gives public/apps/cuda/11.1/bin/nvcc 
export CUDA_HOME=/public/apps/cuda/11.1
python setup.py install
cd ..

