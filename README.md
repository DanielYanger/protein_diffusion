## Application of DDPO in Optimizing DNA sequences for Translational Efficency. 

### Things I've Tried
- Building a diffusion model to generate the coding region
    + Diffusion model has 3 channels, each one representing a codon. Randomly generate a bunch of DNA sequences as the training set and have the diffusion model learn how to generate correct sequences.
    + The diffusion model could be correctly trained to produce valid DNA sequences about ~90% of the time. 
    + Issues arise when trying to apply DDPO or other RL reward methods 

- Using maximize guanine as the reward function
    + Used as a proof of concept that DDPO could be applied to protein generation.
    + Reward function counted the number of guanine and returned 0 if it as an incorrect protein
    + Saw a steady rise in the amount of guanine over time but then a sudden drop off
    + Likely due to the discrete nature of the problem space

- Representing the protein with one channel rather than 3
    + Now 1x(3*length of protein). 
    + No noticable differences in performance 

- Representing a base using a probability distribution
    + Represent the protein using 4 channels for each base. Each channel represents the probability of choosing that specific base. Suggested by William and Alan
    + I took the argmax of the channels and then tried to train on that but this did not work. 
        * Lack of differentiablity
    + Should be taking the softmax and then taking the MSE loss if trying to train the diffusion model

- Using Stable Diffusion and Hugging Face's model as a base
    + There is a prebuilt DDPO library by Hugging Face but it is built upon 2D Conditional Unets
    + Tried to refactor the code and use 1D and non-conditional Unets
    + Some success but very messy code and not better than using Lucid Rain's model and own DDPO

- Diffuse entire protein including UTRs
    + Since diffusion models need a fixed length, encode it as a 1xN tensor
    + Pass in the UTR3 and UTR5 length
    + Couldn't get the model to learn that only the coding region matters
        - Saw some success when tuning the dimensions of the model. Saw the model learn about 50% of the sequence

- Using LGBM as the reward with fixed UTRs and generated coding region
    + Didn't see a lot of success. Ran into similar issues as previous attempts with maxing guanine. 
    + Used EEF2 UTR regions
    + Problem was the model would quickly no longer generate correct proteins

- Using LGBM as reward with fixed coding region and generated UTRs
    + Saw reward increase over time. However, it seems to just converge to UTRs of all T
    + Tried starting at different points, all result in the same behavior
    + Found a bug related to the single step DDIM sampling with log prob. Passed in the next time step instead of the current time step for determining the predicted output

- Using an untrained diffusion model and running DDPO
    + Ran into issues with generating negative values and breaking
    + Softmax/argmax method might allow for this and handle negative numbers

### Things to Do
- Retry the probability distribution
    + Use a softmax or don't use it for training, just directly use it in DDPO

- When DDPO is working
    + Check the diversity of the generated sequences
        * Make sure it isn't all just Ts
    + Try starting from different points to see if it results in different outputs
    + Finetune hyperparameters
    + Try on a known/different protein
    + Try on a shorter protein

- Try DPOK

### Code
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
