import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from src.model.double_convolution import DoubleConv


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super(UNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        for (in_feature, out_feature) in zip([in_channels] + features, features):
            self.encoder.append(DoubleConv(in_feature, out_feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(
                feature * 2, feature, kernel_size=2, stride=2
            ))
            self.decoder.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        encoder_output = x
        for layer in self.encoder:
            encoder_output = layer(encoder_output)
            skip_connections.append(encoder_output)
            encoder_output = self.pooling_layer(encoder_output)

        decoder_output = self.bottleneck(encoder_output)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            decoder_output = self.decoder[idx](decoder_output)
            skip_connection = skip_connections[idx // 2]

            if decoder_output.size() != skip_connection.size():
                decoder_output = TF.resize(decoder_output, size=skip_connection.size()[2:])

            concat_skip = torch.cat((skip_connection, decoder_output), dim=1)
            decoder_output = self.decoder[idx + 1](concat_skip)

        out = self.final_conv(decoder_output)
        return out


def demo():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    print(model)
    preds = model(x)
    print(preds.size(), x.shape)


if __name__ == '__main__':
    demo()
