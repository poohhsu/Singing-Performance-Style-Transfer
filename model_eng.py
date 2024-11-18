import math

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

        self.chs_grp_eng = args.chs_grp_eng
        self.dim_neck_eng = args.dim_neck_eng
        self.dim_enc_eng = args.dim_enc_eng
        self.freq_eng = args.freq_eng
        self.eng_bin = args.eng_bin

        self.conv = nn.Sequential(
            ConvNorm(self.eng_bin, self.dim_enc_eng, kernel_size=11, stride=1, dilation=1, w_init_gain='relu'),
            nn.GroupNorm(self.dim_enc_eng//self.chs_grp_eng, self.dim_enc_eng),
            nn.ReLU()
        )
        self.layer1 = ResBlock(self.dim_enc_eng, self.dim_enc_eng, self.chs_grp_eng, kernel_size=5, dilation=1)
        self.layer2 = ResBlock(self.dim_enc_eng, self.dim_enc_eng, self.chs_grp_eng, kernel_size=5, dilation=1)
        self.layer3 = ResBlock(self.dim_enc_eng, self.dim_enc_eng, self.chs_grp_eng, kernel_size=5, dilation=1)
        self.layer4 = ResBlock(self.dim_enc_eng, self.dim_enc_eng, self.chs_grp_eng, kernel_size=5, dilation=1)
        
        self.lstm = nn.LSTM(self.dim_enc_eng, self.dim_neck_eng, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x_forward = x[:, :, :self.dim_neck_eng]
        x_backward = x[:, :, self.dim_neck_eng:]
        codes = []
        for i in range(0, x.size(1), self.freq_eng):
            codes.append(torch.cat((x_forward[:, min(i+self.freq_eng-1, x.size(1)-1), :], x_backward[:, i, :]), dim=-1))

        return codes

        
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim_neck_eng = args.dim_neck_eng
        self.dim_emb_eng = args.dim_emb_eng
        self.dim_dec_eng = args.dim_dec_eng
        self.chs_grp_eng = args.chs_grp_eng
        self.eng_bin = args.eng_bin

        self.conv = nn.Sequential(
            ConvNorm(self.dim_neck_eng*2 + self.dim_emb_eng + 1, self.dim_dec_eng, kernel_size=11, stride=1, dilation=1, w_init_gain='relu'),
            nn.GroupNorm(self.dim_dec_eng//self.chs_grp_eng, self.dim_dec_eng),
            nn.ReLU()
        )
        self.layer1 = ResBlock(self.dim_dec_eng, self.dim_dec_eng, self.chs_grp_eng, kernel_size=5, dilation=1)
        self.layer2 = ResBlock(self.dim_dec_eng, self.dim_dec_eng, self.chs_grp_eng, kernel_size=5, dilation=1)
        self.layer3 = ResBlock(self.dim_dec_eng, self.dim_dec_eng, self.chs_grp_eng, kernel_size=5, dilation=1)
        self.layer4 = ResBlock(self.dim_dec_eng, self.dim_dec_eng, self.chs_grp_eng, kernel_size=5, dilation=1)
        
        self.lstm = nn.LSTM(self.dim_dec_eng, self.dim_dec_eng, 1, batch_first=True, bidirectional=True)
        
        self.linear_projection = LinearNorm(self.dim_dec_eng*2, self.eng_bin)

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


class EC(nn.Module):
    """Generator network."""
    def __init__(self, args):
        super(EC, self).__init__()
        
        self.args = args
        self.singer_embed = nn.Embedding(300, args.dim_emb_eng)
        torch.nn.init.normal_(self.singer_embed.weight, 0.0, args.dim_emb_eng ** -0.5)

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x_real, c_trg, f0_real):
        x = self.energy_to_coarse(x_real)

        codes = self.encoder(x.transpose(1, 2))
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, self.args.freq_eng, -1))
        code_exp = torch.cat(tmp, dim=1)[:, :x.size(1), :]
        
        c_trg = self.singer_embed(c_trg)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1, x.size(1), -1), f0_real.unsqueeze(-1).log2()), dim=-1)
        
        decoder_outputs = self.decoder(encoder_outputs)

        return x, torch.clamp(torch.sigmoid(decoder_outputs), 1e-10, 1-1e-10)

    def energy_to_coarse(self, eng):
        eng = eng.log10()
        interval = (math.log10(self.args.eng_max) - math.log10(self.args.eng_min)) / self.args.eng_bin
        idx = ((eng - math.log10(self.args.eng_min)) // interval).long()
        weight = 1 - (eng - math.log10(self.args.eng_min) - idx * interval) / interval
        eng_coarse = torch.zeros(eng.size(0), eng.size(1), self.args.eng_bin).to(eng.device)
        eng_coarse[torch.arange(eng.size(0))[:, None], torch.arange(eng.size(1)), torch.clamp(idx, max=self.args.eng_bin-1)] = weight
        eng_coarse[torch.arange(eng.size(0))[:, None], torch.arange(eng.size(1)), torch.clamp(idx + 1, max=self.args.eng_bin-1)] = 1 - weight
        return eng_coarse.float().requires_grad_(False)

    def reverse_embedding(self, outputs):
        engs = []
        interval = (math.log10(self.args.eng_max) - math.log10(self.args.eng_min)) / self.args.eng_bin
        for i in range(outputs.size(0)):
            output = outputs[i].clone()

            eng_mapping = torch.linspace(math.log10(self.args.eng_min), math.log10(self.args.eng_max) - interval, self.args.eng_bin)
            eng_mapping = eng_mapping[None, :].to(outputs.device)
            eng = (eng_mapping * output).sum(dim=1) / output.sum(dim=1)
            engs.append(10 ** eng)
        return torch.stack(engs)