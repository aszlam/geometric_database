# WSNeF

**WARNING** This will only work in the FAIR cluster since that's where all our datasets are, some paths are even hardcoded.

1. Clone the git repo with all the submodules included (only if you are in the private LSeg and Detic forks.) `git clone --recursive git@github.com:aszlam/geometric_database.git`
2. Create a new conda environment, install `pytorch` and [habitat](https://github.com/facebookresearch/habitat-sim#recommended-conda-packages).   
        a. Important to keep the python version <= 3.8.  
        b. Keep python version <= 1.8.2
        b. Important to install pytorch from conda, my pytorch installation from PIP would freeze and crash sometimes.
3. Install the pip requirements, `pip install -r requirements.txt`
4. Log in to wandb, optionally, if you want to track the trainings.
5. Create a `.cache` directory for caching dataloader (speeds up subsequent runs).
5. Run `python train_surface_model.py` to train the implicit models. See `configs/train.yaml` to figure out the configs.


## Create conda env and test

example environment creation

You need ipykernel for adding to jupyter notebook

```bash
conda create -n geom python=3.7
conda activate geom
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia \
 && conda install -y habitat-sim headless withbullet -c aihabitat \
 && conda install -y ipykernel 

pip install -U sentence-transformers
pip install openai-clip
pip install numpy-quaternion
pip install pytorch-lightning==1.3.5
pip install torchmetrics==0.6.0
pip install opencv-python
pip install pyntcloud==0.3.1

# Encoding for lang-seg
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
# Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Set paths to detic or lang-seg:
```
source set_paths.sh
# or just add them to your .bashrc or something
```

Then you can run:
```
conda activate geom
source set_paths.sh  # helper if running from repo root
python train_surface_model.py
```
