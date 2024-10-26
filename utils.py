from PIL import Image
from torch import autocast



def image_grid(imgs, rows, cols):
  assert len(imgs) == rows * cols

  w, h = imgs[0].size
  grid = Image.new('RGB', size=(cols * w, rows * h))

  for i, img in enumerate(imgs):
    grid.paste(img, box=(i % cols * w, i // cols * h))
  return grid


def get_image(filename):
  return Image.open(filename).convert('RGB')


def generate_samples(pipe, prompt, n_columns=1, n_rows=1):
  all_images = []
  for _ in range(n_rows):
    with autocast('cuda'):
      # num_inference_steps changed to 30 due to memory alloc reasons!!
      # it should be 50 by a default
      images = pipe([prompt] * n_columns, num_inference_steps=30, guidance_scale=7.5).images
      all_images.extend(images)

  grid = image_grid(all_images, n_columns, n_rows)
  return grid
