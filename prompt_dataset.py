from torch.utils.data import Dataset


class PromptDataset(Dataset):
  def __init__(self, prompt, num_samples):
    self.prompt = prompt
    self.num_samples = num_samples

  def __len__(self):
    return self.num_samples

  def __getitem__(self, index):
    example = {}
    example["prompt"] = self.prompt
    example["index"] = index
    return example
