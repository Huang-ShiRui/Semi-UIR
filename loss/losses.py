import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()

    def forward(self, xs, ys):
        L2_temp = 0.2 * self.L2(xs, ys)
        L1_temp = 0.8 * self.L1(xs, ys)
        L_total = L1_temp + L2_temp
        return L_total


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, res, gt):
        res = (res + 1.0) * 127.5
        gt = (gt + 1.0) * 127.5
        r_mean = (res[:, 0, :, :] + gt[:, 0, :, :]) / 2.0
        r = res[:, 0, :, :] - gt[:, 0, :, :]
        g = res[:, 1, :, :] - gt[:, 1, :, :]
        b = res[:, 2, :, :] - gt[:, 2, :, :]
        p_loss_temp = (((512 + r_mean) * r * r) / 256) + 4 * g * g + (((767 - r_mean) * b * b) / 256)
        p_loss = torch.mean(torch.sqrt(p_loss_temp + 1e-8)) / 255.0
        return p_loss


class PerpetualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)


# Charbonnier loss
class CharLoss(nn.Module):
    def __init__(self):
        super(CharLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        diff = torch.add(pred, -target)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
