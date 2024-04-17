from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

import os
import datetime
import torch
import tqdm
import time

from absl import flags
from ml_collections import config_flags
from functools import partial
from collections import defaultdict

from DDPO.ddpo_config import get_config
from Diffusion.gaussian_diffusion import GaussianDiffusion1D
from Protein.protein_utrs_replace import Protein_UTR


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "ddpo_config.py", "Training configuration.")

logger = get_logger(__name__)
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


# DDPO Trainer based on https://github.com/kvablack/ddpo-pytorch/blob/main/scripts/train.py
def main():
    config = get_config()
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name += "_" + unique_id

    inference_steps = config.sample.inference_steps

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * inference_steps,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    diffusion = GaussianDiffusion1D.load_diffusion(config.pretrained.model, accelerator.device)

    optimizer = torch.optim.AdamW(diffusion.parameters(),
                                lr=config.train.learning_rate,
                                betas=(config.train.adam_beta1, config.train.adam_beta2),
                                weight_decay=config.train.adam_weight_decay,
                                eps=config.train.adam_epsilon,)
    
    protein_seq = config.sequence
    protein_obj = Protein_UTR(protein_seq, 497, 83)

    reward_fn = protein_obj.scaled_reward_TE_prediction
    autocast = accelerator.autocast

    diffusion, optimizer = accelerator.prepare(diffusion, optimizer)

    samples_per_epoch = config.sample.batch_size* accelerator.num_processes* config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size* accelerator.num_processes* config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    first_epoch = 0
    global_step = 0

    for epoch in range(first_epoch, config.num_epochs):
        diffusion.eval()
        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            with autocast():
                images, latents, log_probs, timesteps = diffusion.ddim_sample(
                    None,
                    batch_size=config.sample.batch_size,
                    eta = config.sample.eta
                )
            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1) 
            timesteps = timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)
            rewards = reward_fn(images)

            samples.append(
                {
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
        accelerator.log(
                    {
                        "reward": rewards,
                        "epoch": epoch,
                        "reward_mean": rewards.mean(),
                        "reward_std": rewards.std(),
                    },
                    step=global_step,
        )

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        
        accelerator.log(
            {"epoch": epoch, "reward": rewards, "advantages": advantages, "reward_mean": rewards.mean(), "reward_std": rewards.std()},
            step=global_step,
        )

        print(f"epoch: {epoch}, reward: {rewards}, advantages: {advantages}, reward_mean: {rewards.mean()}, reward_std: {rewards.std()}")
        

        samples["advantages"] = torch.as_tensor(advantages).reshape(accelerator.num_processes, -1)[accelerator.process_index].to(accelerator.device)

        del samples["rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch)
        assert num_timesteps == config.sample.inference_steps

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            perms = torch.stack([torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)])
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][torch.arange(total_batch_size, device=accelerator.device)[:, None],perms,]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            diffusion.train()
            info = defaultdict(list)

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                for j in tqdm(
                    range(inference_steps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(diffusion):
                        with autocast():
                            # TODO: Verify this
                            # noise_pred = diffusion.model(
                            #     sample["latents"][:, j],
                            #     sample["timesteps"][:, j],
                            # )

                            log_prob = diffusion.ddim_single_step(
                                config.train.batch_size,
                                sample["latents"][:, j],
                                sample["timesteps"][:, j],
                                sample["next_latents"][:, j],
                                config.sample.eta,
                            )
                            
                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(diffusion.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        assert (j == inference_steps - 1) and (i + 1) % config.train.gradient_accumulation_steps == 0
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        print(info)
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            updated_diffusion = accelerator.unwrap_model(diffusion)
            updated_diffusion.save_model(config.results_folder, milestone=epoch//config.save_freq)
            torch.save(images, f"{config.results_folder}/sample-{epoch//config.save_freq}.pt")

if __name__ == "__main__":
    main()