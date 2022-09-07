# WSNeF

**WARNING** This will only work in the FAIR cluster since that's where all our datasets are, some paths are even hardcoded.

1. Clone the git repo with all the submodules included (only if you are in the private LSeg and Detic forks.) `git clone --recursive git://github.com/aszlam/geometric_database.git`
2. Create a new conda environment, install `pytorch` and [habitat](https://github.com/facebookresearch/habitat-sim#recommended-conda-packages).   
        a. Important to keep the python version <= 3.8.  
        b. Important to install pytorch from conda, my pytorch installation from PIP would freeze and crash sometimes.
3. Install the pip requirements, `pip install -r requirements.txt`
4. Log in to wandb, optionally, if you want to track the trainings.
5. Create a `.cache` directory for caching dataloader (speeds up subsequent runs).
5. Run `python train_surface_model.py` to train the implicit models. See `configs/train.yaml` to figure out the configs.
