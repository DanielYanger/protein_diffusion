## Application of DDPO in Optimizing DNA sequences for Translational Efficency. 

### Training Models
- Install the dependencies from `requirements.txt`
- Diffusion
    + `train_diffusion_utrs.py` and `train_diffusion.py` are the files to run
    + Run the command `accelerate launch file` to start the training process. Hyperparameters can be tuned in the file
    + The type of diffusion model (only coding region, full sequence, fixed coding region, etc.) is dictated by the `Protein` class.
    + `TACC Scripts/diffusion.bash` has an example script that can be used with the TACC LS6 machines. 
- DDPO
    + `ddpo_trainer_replace.py` and `ddpo_trainer.py` are the scripts to run
    + Run the command `accelerate launch file` to start the training process. 
    + Hyperparameters are tunable in `DDPO/ddpo_config.py`. If you need multiple profiles, copy the file and change the file referenced in the training scripts. 
        - Most important hyper parameters are:
            - `config.num_epochs`: Number of epochs to train on
            - `config.results_folder`: Location to save the model and samples to
            - `config.sequence`: Protein sequence used
            - `pretrained.model`: Folder containing the pretrained diffusion model. Note, it doesn't have to be well trained but it should be somewhat trained for some reason. More investigation needs to go into this.
            - `sample.batch_size`: Sequences to generate per batch
            - `sample.num_batches_per_epoch`: Number of batches to generate per epoch. Total effective size is # GPUs * batch_size * num_batches_per_epoch.
            - `train.batch_size`: Batch size to use for training. Needs to evenly divide total effective size
            - `train.learning_rate`: Learning rate
            - `train.adv_clip_max`: Maximum allowed advantage, which is the standardized reward. Basically prevents a single sample from driving the model's direction
            - `train.clip_range`: Maximum deviation from the original log ratio. Used in the PPO logic. 


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

- Debugging DDPO
    + Spent a lot of time comparing things to fully understand what was going wrong
    + Miscomputation of the log prob resulted in incorrect training values. Essentially, the log probs should be equal during the first training epoch but then once it's been trained, they start to differ a bit. This should be somewhat reflected in the kl divergence and clip ratio.

- DDPO Rewards
    + Finetuning the reward is very important. For example, the model initially converged to the UTRs being all T. However, once I changed the reward to penalize not being diverse enough (over 50% of a single base), it started to produce UTR5 regions that were diverse and UTR3 regions that contained long strings of just T's but still having some diversity. 
    + Penalizing certain behavior is completely possible to encourage or discourage certain patterns.

### Long Term Objectives
- Retry the probability distribution
    + Use a softmax or don't use it for training, just directly use it in DDPO
    + Potentially with the right hyperparameters you can train the model to produce the entire sequence (coding regions + utrs) without needing to replace with a fixed value. The hyperparameter that seemed to do the most was the dimensions. You can have a list of length no greater than the number of factors of 2 plus one in the length of the sequence. So for example, if the length is 3780 ($2^2*3^3*5*7$), you can only have 3 additional dimensions. Tuning these seem to change the most as I got a model to learn the first part of the sequence. 

- DDPO Related Improvements
    + Try starting from different points to see if it results in different outputs
        + Change the seed in the config file
    + Finetune hyperparameters
        + Increase Batch Size, tune learning rate, clip range, max advantage
    + Try on a known protein
        + See if we can get an actual improvement
    + Understand why it breaks with untrained model
        + Might already be fixed but untested

- DPOK
    + Implement to see if there are any differences

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
