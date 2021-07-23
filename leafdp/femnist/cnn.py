import torch
import torch.nn as nn
import torch.nn.functional as F


class FemnistModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 28,
        filters: tuple = 32,
        kernel_size: int = 5,
        num_classes: int = 62,
        pad: int = 0,
    ):
        super(FemnistModel, self).__init__()
        self.input_dim = input_dim
        # Best number in the GroupNorm paper
        channels_per_group = 16
        self.conv1 = nn.Conv2d(1, filters, kernel_size, 1, padding=pad)
        self.gn1 = nn.GroupNorm(int(filters / channels_per_group), filters)
        self.tanh1 = nn.Tanh()
        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size, 1, padding=pad)
        self.gn2 = nn.GroupNorm(int((filters * 2) / channels_per_group), filters * 2)
        self.tanh2 = nn.Tanh()
        im_size = int((self.input_dim - 12 + 12 * pad) / 4)
        self.linear = nn.Linear(filters * 2 * im_size ** 2, num_classes)

    def forward(self, X: torch.Tensor):
        batch_size = X.shape[0]
        out = X.view(batch_size, 1, self.input_dim, self.input_dim)
        out = self.conv1(out)
        out = self.gn1(out)
        out = self.tanh1(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.tanh2(out)
        out = F.avg_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

    def predict(self, X: torch.Tensor):
        return F.softmax(self.forward(X))
