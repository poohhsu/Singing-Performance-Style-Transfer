import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, chs_grp, kernel_size=5, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        padding = int(dilation * (kernel_size - 1) / 2)
        self.left = nn.Sequential(
            ConvNorm(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, w_init_gain='relu'),
            nn.GroupNorm(output_channels//chs_grp, output_channels),
            nn.ReLU(),
            ConvNorm(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, w_init_gain='relu'),
            nn.GroupNorm(output_channels//chs_grp, output_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                ConvNorm(input_channels, output_channels, kernel_size=1, stride=stride, dilation=dilation, w_init_gain='relu'),
                nn.GroupNorm(output_channels//chs_grp, output_channels),
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.chs_grp = args.chs_grp
        self.dim_neck = args.dim_neck
        self.dim_enc = args.dim_enc
        self.freq = args.freq

        self.conv = nn.Sequential(
            ConvNorm(72, self.dim_enc, kernel_size=11, stride=1, dilation=1, w_init_gain='relu'),
            nn.GroupNorm(self.dim_enc//self.chs_grp, self.dim_enc),
            nn.ReLU()
        )
        self.layer1 = ResBlock(self.dim_enc, self.dim_enc, self.chs_grp, kernel_size=5, dilation=1)
        self.layer2 = ResBlock(self.dim_enc, self.dim_enc, self.chs_grp, kernel_size=5, dilation=1)
        self.layer3 = ResBlock(self.dim_enc, self.dim_enc, self.chs_grp, kernel_size=5, dilation=1)
        self.layer4 = ResBlock(self.dim_enc, self.dim_enc, self.chs_grp, kernel_size=5, dilation=1)
        
        self.lstm = nn.LSTM(self.dim_enc, self.dim_neck, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x_forward = x[:, :, :self.dim_neck]
        x_backward = x[:, :, self.dim_neck:]
        codes = []
        for i in range(0, x.size(1), self.freq):
            codes.append(torch.cat((x_forward[:, min(i+self.freq-1, x.size(1)-1), :], x_backward[:, i, :]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim_neck = args.dim_neck
        self.dim_emb = args.dim_emb
        self.dim_dec = args.dim_dec
        self.chs_grp = args.chs_grp

        self.conv = nn.Sequential(
            ConvNorm(self.dim_neck*2 + self.dim_emb, self.dim_dec, kernel_size=11, stride=1, dilation=1, w_init_gain='relu'),
            nn.GroupNorm(self.dim_dec//self.chs_grp, self.dim_dec),
            nn.ReLU()
        )
        self.layer1 = ResBlock(self.dim_dec, self.dim_dec, self.chs_grp, kernel_size=5, dilation=1)
        self.layer2 = ResBlock(self.dim_dec, self.dim_dec, self.chs_grp, kernel_size=5, dilation=1)
        self.layer3 = ResBlock(self.dim_dec, self.dim_dec, self.chs_grp, kernel_size=5, dilation=1)
        self.layer4 = ResBlock(self.dim_dec, self.dim_dec, self.chs_grp, kernel_size=5, dilation=1)
        
        self.lstm = nn.LSTM(self.dim_dec, self.dim_dec, 1, batch_first=True, bidirectional=True)
        
        self.linear_projection = LinearNorm(self.dim_dec*2, 72)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        
        x = self.linear_projection(x)

        return x
    

class PC(nn.Module):
    """Generator network."""
    def __init__(self, args):
        super(PC, self).__init__()
        
        self.args = args
        self.singer_embed = nn.Embedding(300, args.dim_emb)
        torch.nn.init.normal_(self.singer_embed.weight, 0.0, args.dim_emb ** -0.5)

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x_real, c_trg):
        x = self.f0_to_coarse(x_real)

        codes = self.encoder(x.transpose(1, 2))
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, self.args.freq, -1))
        code_exp = torch.cat(tmp, dim=1)[:, :x.size(1), :]
        
        c_trg = self.singer_embed(c_trg)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        
        decoder_outputs = self.decoder(encoder_outputs)

        return x, torch.sigmoid(decoder_outputs)

    def f0_to_coarse(self, f0):
        cent = 1200 * torch.log2(f0 / 10.)
        idx = ((cent - 2051.1487628680297) // 100).long()
        weight = 1 - (cent - 2051.1487628680297 - idx * 100) / 100
        f0_coarse = torch.zeros(f0.size(0), f0.size(1), 72).to(f0.device)
        f0_coarse[torch.arange(cent.size(0))[:, None], torch.arange(cent.size(1)), idx] = weight
        f0_coarse[torch.arange(cent.size(0))[:, None], torch.arange(cent.size(1)), torch.clamp(idx + 1, max=71)] = 1 - weight
        return f0_coarse.float().requires_grad_(False)

    def reverse_embedding(self, outputs):
        f0s = []
        for i in range(outputs.size(0)):
            output = outputs[i].clone()

            cents_mapping = torch.linspace(0, 7100, 72) + 2051.1487628680297
            cents_mapping = cents_mapping[None, :].to(outputs.device)
            cent = (cents_mapping * output).sum(dim=1) / output.sum(dim=1)
            f0s.append(10 * 2 ** (cent / 1200))
        return torch.stack(f0s)