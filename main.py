import os
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from tqdm.auto import tqdm
from pathlib import Path
import gc
import accelerate

from constants import *
from dreambooth import DreamBooth
from params import args
from prompt_dataset import PromptDataset
from utils import get_image, generate_samples


def run(
    input_image_path='data/',
    instance_prompt='a photo of x',
    class_prompt='a photo of human face',
    num_class_images=5,
    acceleration=True
):
  """ Settings for teaching a new concept
  """
  images = list(filter(
    None, [get_image(input_image_path + 'img{}.jpg'.format(i)) for i in range(1, num_class_images + 1)]
  ))
  save_path = "./my_concept"
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  [image.save(f"{save_path}/{i}.jpeg") for i, image in enumerate(images)]

  args.instance_data_dir = save_path
  args.instance_prompt = instance_prompt
  args.class_prompt = class_prompt

  """ Teaching the model a new concept
  """
  if prior_preservation:
    class_images_dir = Path(class_data_root)
    if not class_images_dir.exists():
      class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < num_class_images:
      pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, use_auth_token=True, revision="fp16", torch_dtype=torch.float16
      ).to("cuda")
      pipeline.enable_attention_slicing()
      pipeline.set_progress_bar_config(disable=True)

      num_new_images = num_class_images - cur_class_images
      print(f"Number of class images to sample: {num_new_images}.")

      sample_dataset = PromptDataset(class_prompt, num_new_images)
      sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size)

      for example in tqdm(sample_dataloader, desc="Generating class images"):
        with torch.autocast("cuda"):
          images = pipeline(example["prompt"]).images

        for i, image in enumerate(images):
          image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")
      pipeline = None
      gc.collect()
      del pipeline
      with torch.no_grad():
        torch.cuda.empty_cache()

  model = DreamBooth(pretrained_model_name_or_path)

  if acceleration:
    accelerate.notebook_launcher(model.train, args=(args,))
  else:
    model.train(args)

  with torch.no_grad():
    torch.cuda.empty_cache()

  pipe = StableDiffusionPipeline.from_pretrained(
    args.output_dir,
    torch_dtype=torch.float16,
  ).to("cuda")
  return pipe


# Example usage
# if __name__ == '__main__':
#   pipe = run()
#   generate_samples(pipe)
