from model import common
from common.wavelet import DWT_Haar
import torch.nn as nn

class Discriminator(nn.Module):
    '''
        output is not normalized
    '''
    def __init__(self, args):
        super(Discriminator, self).__init__()
        in_channels = args.n_colors
        out_channels = 64
        patch_size = args.patch_size
        
        self.LLgt = args.LLgt
        
        depth = 5

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))
        # print(patch_size)
        patch_size = patch_size // (2**((depth + 1) // 2))
        # print(patch_size)
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]

        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        if self.adv_wvt:
            x = self.dwt(x)
            # x *= self.weights
        # print(x.size())
        features = self.features(x)
        # print(features.size())
        output = self.classifier(features.view(features.size(0), -1))

        return output
