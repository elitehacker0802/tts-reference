
from abc import ABC

import torch

from Utility.utils import pad_list


class LengthRegulator(torch.nn.Module, ABC):


    def __init__(self, pad_value=0.0):
        """
        Initialize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.
        """
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, alpha=1.0):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).
        """
        if alpha != 1.0:
            assert alpha > 0
            ds = torch.round(ds.float() * alpha).long()

        if ds.sum() == 0:
            ds[ds.sum(dim=1).eq(0)] = 1

        return pad_list([self._repeat_one_sequence(x, d) for x, d in zip(xs, ds)], self.pad_value)

    def _repeat_one_sequence(self, x, d):
        """
        Repeat each frame according to duration
        """
        return torch.repeat_interleave(x, d, dim=0)
