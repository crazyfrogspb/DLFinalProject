import torch
import torch.nn as nn
import torch.nn.functional as F

from dlfinalproject.models.layers import (BasicConv2d, GatedConv2d,
                                          GatedDeconv2d, SpectralConv2d,
                                          get_pad, same_pad)


class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation, with_attn=False):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        if self.with_attn:
            return out, attention
        else:
            return out


class Classifier(torch.nn.Module):
    def __init__(self, n_in_channel=3, batch_norm=False):
        super().__init__()
        cnum = 32
        self.extractor = nn.Sequential(BasicConv2d(
            n_in_channel, cnum, 5, 1, padding=get_pad(96, 5, 1), batch_norm=batch_norm),
            # downsample 128
            BasicConv2d(
            cnum, 2 * cnum, 4, 2, padding=get_pad(96, 4, 2), batch_norm=batch_norm),
            BasicConv2d(
            2 * cnum, 2 * cnum, 3, 1, padding=get_pad(48, 3, 1), batch_norm=batch_norm),
            # downsample to 64
            BasicConv2d(
            2 * cnum, 4 * cnum, 4, 2, padding=get_pad(48, 4, 2), batch_norm=batch_norm),
            BasicConv2d(
            4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            BasicConv2d(
            4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            # atrous convlution
            BasicConv2d(
            4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(24, 3, 1, 2), batch_norm=batch_norm),
            BasicConv2d(
            4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(24, 3, 1, 4), batch_norm=batch_norm),
            BasicConv2d(
            4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(24, 3, 1, 8), batch_norm=batch_norm),
            BasicConv2d(
            4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(24, 3, 1, 16), batch_norm=batch_norm),
            BasicConv2d(
            4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            BasicConv2d(
            4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm))
        self.fc1 = nn.Linear(128 * 24 * 24, 2000)
        self.fc2 = nn.Linear(2000, 1500)
        self.fc3 = nn.Linear(1500, 1000)

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Generator(torch.nn.Module):
    def __init__(self, n_in_channel=5, batch_norm=False):
        super().__init__()
        cnum = 32
        self.coarse_net = nn.Sequential(
            # input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2d(
                n_in_channel, cnum, 5, 1, padding=get_pad(96, 5, 1), batch_norm=batch_norm),
            # downsample 128
            GatedConv2d(
                cnum, 2 * cnum, 4, 2, padding=get_pad(96, 4, 2), batch_norm=batch_norm),
            GatedConv2d(
                2 * cnum, 2 * cnum, 3, 1, padding=get_pad(48, 3, 1), batch_norm=batch_norm),
            # downsample to 64
            GatedConv2d(
                2 * cnum, 4 * cnum, 4, 2, padding=get_pad(48, 4, 2), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            # atrous convlution
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(24, 3, 1, 2), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(24, 3, 1, 4), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(24, 3, 1, 8), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(24, 3, 1, 16), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            # upsample
            GatedDeconv2d(
                2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(48, 3, 1), batch_norm=batch_norm),
            GatedConv2d(
                2 * cnum, 2 * cnum, 3, 1, padding=get_pad(48, 3, 1), batch_norm=batch_norm),
            GatedDeconv2d(
                2, 2 * cnum, cnum, 3, 1, padding=get_pad(96, 3, 1), batch_norm=batch_norm),

            GatedConv2d(
                cnum, cnum // 2, 3, 1, padding=get_pad(48, 3, 1), batch_norm=batch_norm),
            GatedConv2d(
                cnum // 2, 3, 3, 1, padding=get_pad(96, 3, 1), activation=None, use_gates=False, batch_norm=False),
            nn.Tanh()
        )

        self.refine_net = nn.Sequential(
            # input is 5*256*256
            GatedConv2d(
                n_in_channel, cnum, 5, 1, padding=get_pad(96, 5, 1), batch_norm=batch_norm),
            # downsample
            GatedConv2d(
                cnum, cnum, 4, 2, padding=get_pad(96, 4, 2), batch_norm=batch_norm),
            GatedConv2d(
                cnum, 2 * cnum, 3, 1, padding=get_pad(48, 3, 1), batch_norm=batch_norm),
            # downsample
            GatedConv2d(
                2 * cnum, 2 * cnum, 4, 2, padding=get_pad(48, 4, 2), batch_norm=batch_norm),
            GatedConv2d(
                2 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(24, 3, 1, 2), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(24, 3, 1, 4), batch_norm=batch_norm),
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(24, 3, 1, 8), batch_norm=batch_norm),

            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(24, 3, 1, 16), batch_norm=batch_norm)
        )
        self.attn = SelfAttention(4 * cnum, 'relu', with_attn=False)
        self.upsample_net = nn.Sequential(
            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),

            GatedConv2d(
                4 * cnum, 4 * cnum, 3, 1, padding=get_pad(24, 3, 1), batch_norm=batch_norm),
            GatedDeconv2d(
                2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(48, 3, 1), batch_norm=batch_norm),
            GatedConv2d(
                2 * cnum, 2 * cnum, 3, 1, padding=get_pad(48, 3, 1), batch_norm=batch_norm),
            GatedDeconv2d(
                2, 2 * cnum, cnum, 3, 1, padding=get_pad(96, 3, 1), batch_norm=batch_norm),

            GatedConv2d(
                cnum, cnum // 2, 3, 1, padding=get_pad(96, 3, 1), batch_norm=batch_norm),
            GatedConv2d(
                cnum // 2, 3, 3, 1, padding=get_pad(96, 3, 1), activation=None, use_gates=False, batch_norm=False),
            nn.Tanh()
        )

    def forward(self, imgs, masks, img_exs=None):
        # Coarse
        if img_exs is None:
            input_imgs = torch.cat(
                [imgs, torch.full_like(masks, 1.), masks], dim=1)
        else:
            input_imgs = torch.cat(
                [imgs, torch.full_like(masks, 1.), img_exs, masks], dim=1)
        x = self.coarse_net(input_imgs)
        coarse_x = x
        # Refine
        masked_imgs = imgs * (1 - masks) + coarse_x * masks
        if img_exs is None:
            input_imgs = torch.cat(
                [masked_imgs, torch.full_like(masks, 1.), masks], dim=1)
        else:
            input_imgs = torch.cat(
                [masked_imgs, torch.full_like(masks, 1.), img_exs, masks], dim=1)
        x = self.refine_net(input_imgs)
        x = self.attn(x)
        x = self.upsample_net(x)
        return coarse_x, x


class Discriminator(nn.Module):
    def __init__(self, n_in_channel=4, spectral=False, batch_norm=True):
        super().__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SpectralConv2d(n_in_channel, 2 * cnum, 5,
                           2, padding=same_pad(96, 48, 2, 5), spectral=spectral, batch_norm=batch_norm),
            SpectralConv2d(2 * cnum, 4 * cnum, 5, 2,
                           padding=same_pad(48, 24, 2, 5), spectral=spectral, batch_norm=batch_norm),
            SpectralConv2d(4 * cnum, 8 * cnum, 5, 2,
                           padding=same_pad(24, 12, 2, 5), spectral=spectral, batch_norm=batch_norm),
            SpectralConv2d(8 * cnum, 8 * cnum, 5, 2,
                           padding=same_pad(12, 6, 2, 5), spectral=spectral, activation=None, batch_norm=False)
        )

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0), -1))
        return x
