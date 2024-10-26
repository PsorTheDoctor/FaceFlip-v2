import math
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import bitsandbytes as bnb

from dreambooth_dataset import DreamBoothDataset


class DreamBooth:
  def __init__(self, pretrained_model_name_or_path):
    self.text_encoder = CLIPTextModel.from_pretrained(
      pretrained_model_name_or_path, subfolder="text_encoder", use_auth_token=True
    )
    self.vae = AutoencoderKL.from_pretrained(
      pretrained_model_name_or_path, subfolder="vae", use_auth_token=True
    )
    self.unet = UNet2DConditionModel.from_pretrained(
      pretrained_model_name_or_path, subfolder="unet", use_auth_token=True
    )
    self.tokenizer = CLIPTokenizer.from_pretrained(
      pretrained_model_name_or_path,
      subfolder="tokenizer",
      use_auth_token=True,
    )

  def train(self, args):
    logger = get_logger(__name__)

    accelerator = Accelerator(
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      mixed_precision=args.mixed_precision,
    )
    if args.gradient_checkpointing:
      self.unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
      optimizer_class = bnb.optim.AdamW8bit
    else:
      optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
      self.unet.parameters(),  # only optimize unet
      lr=args.learning_rate,
    )
    noise_scheduler = DDPMScheduler(
      beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    train_dataset = DreamBoothDataset(
      instance_data_root=args.instance_data_dir,
      instance_prompt=args.instance_prompt,
      class_data_root=args.class_data_dir if args.with_prior_preservation else None,
      class_prompt=args.class_prompt,
      tokenizer=self.tokenizer,
      size=args.resolution,
      center_crop=args.center_crop,
    )

    def collate_fn(examples):
      input_ids = [example["instance_prompt_ids"] for example in examples]
      pixel_values = [example["instance_images"] for example in examples]

      # concat class and instance examples for prior preservation
      if args.with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

      pixel_values = torch.stack(pixel_values)
      pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
      input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

      batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
      }
      return batch

    train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )
    unet, optimizer, train_dataloader = accelerator.prepare(self.unet, optimizer, train_dataloader)

    # Move text_encode and vae to gpu
    self.text_encoder.to(accelerator.device)
    self.vae.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
      unet.train()
      for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
          # Convert images to latent space
          with torch.no_grad():
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215

          # Sample noise that we'll add to the latents
          noise = torch.randn(latents.shape).to(latents.device)
          bsz = latents.shape[0]
          # Sample a random timestep for each image
          timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
          ).long()

          # Add noise to the latents according to the noise magnitude at each timestep
          # (this is the forward diffusion process)
          noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

          # Get the text embedding for conditioning
          with torch.no_grad():
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

          # Predict the noise residual
          noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

          if args.with_prior_preservation:
            # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

            # Compute prior loss
            prior_loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="none").mean([1, 2, 3]).mean()

            # Add the prior loss to the instance loss.
            loss = loss + args.prior_loss_weight * prior_loss
          else:
            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

          accelerator.backward(loss)
          accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
          optimizer.step()
          optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
          progress_bar.update(1)
          global_step += 1

        logs = {"loss": loss.detach().item()}
        progress_bar.set_postfix(**logs)

        if global_step >= args.max_train_steps:
          break

      accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if args.censorship:
      safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    else:
      def dummy_checker(images, **kwargs):
        return images, False
      safety_checker = dummy_checker

    if accelerator.is_main_process:
      pipeline = StableDiffusionPipeline(
        text_encoder=self.text_encoder,
        vae=self.vae,
        unet=accelerator.unwrap_model(self.unet),
        tokenizer=self.tokenizer,
        scheduler=PNDMScheduler(
          beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
        ),
        safety_checker=safety_checker,
        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
      )
      pipeline.save_pretrained(args.output_dir)
