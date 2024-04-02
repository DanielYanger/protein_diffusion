Application of DDPO in Optimizing DNA sequences for Translational Efficency. 

**Diffusion**
- Basic code for a 1 dimensional gaussian diffusion based on the [Lucid Rain implementaion](1)
    - `Diffusion/diffusion_trainer.py`: Class for training a gaussian diffusion model. 
        - `diffusion_model`: Gaussian Diffusion Model to be trained
        - `dataset`: Dataset containing data to train on
    - `Diffusion/gaussian_diffusion.py`: Class representing the Gaussian Diffusion Model.
        - `model`: The underlying Unet1D to train.
    - `Diffusion/modules_1D.py`: Class containing various helper functions
        - `Dataset1D`: Class encapsulating the dataset
    - `Diffusion/unet_1d.py`: Class containing the UNet model. 
        - `dim_mults`: Dimensions to be used. Each dimension other than the final will reduce the length by 2.
        - `channels`: Number of channels

**DDPO**
- Code for performing DDPO on a one dimensional Gaussian Diffusion. Based on the [DDPO Paper](2)
    - `DDPO/ddpo_config.py`: Config file for performing DDPO.
        - `config.num_epochs`: Number of epochs to train the model for
        - `config.save_freq`: How often to save
        - `config.results_folder`: Where to save the model
        - `config.sequence`: DNA sequence to train the model
        - `pretrained.model`: Pretrained Gaussian Diffusion Model

**Protein**
- Code for generating and checking protein sequences.
    - `Protein/protein_utrs_replace.py`: Protein class that generates utrs and replaces the coding region
    - `Protein/protein_utrs.py`: Protein class that generates utrs and coding region
    - `Protein/protein.py`: Protein class tha only generates the coding region
- Methods:
    - `multi_check_sequence`: Takes in an array of sequences and checks to see if they match the protein
    - `reward_TE_prediction`: Computes the reward (multiplied by a scale factor) based on the LGBM Model.
    - `generate_n_sequences`: Generates `n` random sequences that encode for the protein

**LGBM**
- Light GBM model for predicting TE of a sequence
    - `LGBM/lgbm.py`: LGBM model that takes in a coding region and uses fixed UTRs
    - `LGBM/lgbm_utr.py`: LGBM model that takes in a full sequence and computes the TE

**TACC Scripts**
- Scripts for launching jobs on TACC
    - `TACC Scripts/ddpo.bash`: Bash script for running DDPO on TACC
    - `TACC Scripts/diffusion.bash`: Bash script for training gaussian diffusion on TACC

**DDPO Training Scripts**
- Training scripts for running DDPO
    - `ddpo_trainer.py`: DDPO trainer for training using a Gaussian diffusion model that only generates coding region
    - `ddpo_trainer_replace.py`: DDPO trainer for training a Gaussian diffusion model that generates the entire sequence but replaces the coding reigon before computing the reward

**Diffusion Training Scripts**
- Training scripts for training Gaussian Diffusion
    - `train_diffusion_utrs.py`: Script for training a Gaussian diffusion model that generates full sequence
    - `train_diffusion.py`: Script for training a Gaussian diffusion model that only generates the coding region

[1]: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py
[2]: https://rl-diffusion.github.io/
