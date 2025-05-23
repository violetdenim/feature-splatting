import torch
import torch.nn as nn
import torch.nn.functional as F

class two_layer_mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim_dict):
        super(two_layer_mlp, self).__init__()
        self.hidden_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        feature_branch_dict = {}
        for key, feat_dim_chw in feature_dim_dict.items():
            feature_branch_dict[key] = nn.Conv2d(hidden_dim, feat_dim_chw[0], kernel_size=1, stride=1, padding=0)
        self.feature_branch_dict = nn.ModuleDict(feature_branch_dict)

    def forward(self, x):
        intermediate_feature = self.hidden_conv(x)
        intermediate_feature = F.relu(intermediate_feature)
        ret_dict = {}
        for key, nn_mod in self.feature_branch_dict.items():
            ret_dict[key] = nn_mod(intermediate_feature)
        return ret_dict
    
    @torch.no_grad()
    def per_gaussian_forward(self, x):
        intermediate_feature = F.linear(x, self.hidden_conv.weight.view(self.hidden_conv.weight.size(0), -1), self.hidden_conv.bias)
        intermediate_feature = F.relu(intermediate_feature)
        ret_dict = {}
        for key, nn_mod in self.feature_branch_dict.items():
            ret_dict[key] = F.linear(intermediate_feature, nn_mod.weight.view(nn_mod.weight.size(0), -1), nn_mod.bias)
        return ret_dict


class cnn_decoder(nn.Module):
    def __init__(self, input_dim, feature_dim_dict):
        super().__init__()
        # self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()
        feature_branch_dict = {}
        for key, feat_dim_chw in feature_dim_dict.items():
            feature_branch_dict[key] = nn.Conv2d(input_dim, feat_dim_chw[0], kernel_size=1, stride=1, padding=0)
        self.feature_branch_dict = nn.ModuleDict(feature_branch_dict)

    def forward(self, x):
        # return self.conv(x)
        ret_dict = {}
        for key, nn_mod in self.feature_branch_dict.items():
            ret_dict[key] = nn_mod(x)
        return ret_dict
    
    @torch.no_grad()
    def per_gaussian_forward(self, x):
        ret_dict = {}
        for key, nn_mod in self.feature_branch_dict.items():
            ret_dict[key] = F.linear(x, nn_mod.weight.view(nn_mod.weight.size(0), -1), nn_mod.bias)
        return ret_dict