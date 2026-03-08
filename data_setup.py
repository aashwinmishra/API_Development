import torch
import numpy as np


class PtychoDataset(Dataset):
  """
  Defines a PyTorch dataset for FNO models. The input to the model is has 3 channels and the
  output has 2 chanels.
  Attributes:
    inputs: Inputs to the model of shape [N, S, S, 3], where the channels are the
            diffraction pattern, local x grid point, local y grid point.
    targets: Model targets of shape [N, S, S, 2] where the channels are the
            amplitude and the phase over the grid.
  """
  def __init__(self,
               x_path,
               intensity_target_path,
               phase_target_path):
    """
    Initializes an instance of the Ptycho Dataset for the FNO Model.
    Args:
      x_path: path to npy file with the PtychoNN inputs of shape [N, 1, S, S].
      intensity_target_path: path to npy file with the intensity targets of shape [N, 1, S, S].
      phase_target_path: path to npy file with the phase targets of shape [N, 1, S, S].
    """
    x = torch.from_numpy(np.load(x_path)).to(torch.float)                                   #[N, 1, S, S]
    y_int = torch.from_numpy(np.load(intensity_target_path)).to(torch.float)                #[N, 1, S, S]
    y_phi = torch.from_numpy(np.load(phase_target_path)).to(torch.float)                    #[N, 1, S, S]
    self.inputs = x                                                                         #[N, 1, S, S]
    self.targets = torch.cat([y_int, y_phi], dim=1)                                         #[N, 2, S, S]

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx: int):
    return {
            'x': self.inputs[idx],    # Input diffraction pattern
            'y': self.targets[idx]    # Target Amplitude & Phase
        }

