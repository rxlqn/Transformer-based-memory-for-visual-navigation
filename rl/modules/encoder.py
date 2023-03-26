import torch
from torch import device, nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.pooling import MaxPool2d
import torch.utils.model_zoo as model_zoo
import torchvision
# from resnet import ResNet50
# from model import Depth_encoding_Net
import kornia
import torchvision.transforms as transforms
import numpy as np

from rl.modules.SMT import SMT_state_encoder

# 图像增强
aug_trans = nn.Sequential(
    nn.MaxPool2d(kernel_size=2),
    nn.ReplicationPad2d(8),
    kornia.augmentation.RandomCrop((90,160))    ## 有点慢 drqv2
)

# rgb2gray = kornia.color.RgbToGrayscale()        ## (N,3,H,W) -> (N,1,H,W)

def weights_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):  
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, feature_dim):
        super(Encoder, self).__init__()
        # assert len(obs_shape) == 3
        self.num_layers = 6
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs_rgb = nn.ModuleList([
            nn.Conv2d(3, self.num_filters, 3, stride=2),                    ## rgb 
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.convs_Depth = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, 3, stride=2),                    ## d
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(25792, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

        # self.apply(weights_init_)       ## 初始化


    def forward_conv_rgb(self, obs):
        self.outputs['rgb'] = obs

        conv = torch.relu(self.convs_rgb[0](obs))
        self.outputs['conv1_g'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs_rgb[i](conv))

            self.outputs['conv_g%s' % (i + 1)] = conv

        h = conv.reshape(conv.size(0), -1)
        return h

    def forward_conv_depth(self, obs):
        self.outputs['depth'] = obs

        conv = torch.relu(self.convs_Depth[0](obs))
        self.outputs['conv1_d'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs_Depth[i](conv))
            self.outputs['conv_d%s' % (i + 1)] = conv

        h = conv.reshape(conv.size(0), -1)
        return h

    def forward(self, rgb, depth, task_obs):

        ## 训练时执行图像增强
        rgb = aug_trans(rgb)
        depth = aug_trans(depth)
        h1 = self.forward_conv_rgb(rgb)
        h2 = self.forward_conv_depth(depth)

        h = torch.cat((h1,h2),1)
 
        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        out = torch.cat((out, task_obs),axis = 1)
        self.outputs['out'] = out

        return out


class Trans_Encoder(nn.Module):
    """CNN encoder followed by a SMT encoder"""
    def __init__(self):
        super(Trans_Encoder, self).__init__()
        self.encoder = Encoder(feature_dim=252)
        self.smt_encoder = SMT_state_encoder(d_model=256, nhead=4)

    def forward(self, rgb, depth, task_obs, key_padding_mask):
        '''
        input: 
                embeddings: from 0 to t
                observations: from 0 to t
                batch = 1
        considering update
        output:
                T D
        '''
        # T N H W C -> T N C H W    64*2*D    TND  之前batch就是seqence，现在包含多个环境的信息，需要flatten 成 TN-》N
        seq_l, batch_size = task_obs.size(0), task_obs.size(1)
        rgb = rgb.flatten(0,1).permute(0, 3, 1, 2)
        depth = depth.flatten(0,1).permute(0, 3, 1, 2)
        task_obs = task_obs.flatten(0,1)
        ## T N D
        embeddings = self.encoder(rgb, depth, task_obs).reshape(seq_l, batch_size, -1)

        ## T N D
        ## 训练，src 的memory和tgt相同都是embedding
        out = self.smt_encoder(o = embeddings, M = embeddings, flag = 1, key_padding_mask = key_padding_mask)

        # T N D
        out = torch.cat((out, task_obs.reshape(seq_l, batch_size, -1)), -1)

        return out
        
    def inference_forward(self, rgb, depth, task_obs, memory, key_padding_mask):
        # N H W C -> N C H W
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)

        ## T N D
        embeddings = self.encoder(rgb, depth, task_obs).unsqueeze(0)

        # 探索，src 变成memory，tgt是当前推理出来的embedding
        if len(memory) < 32:    ## 窗口加快训练速度和收敛速度
            memory = torch.cat((memory, embeddings), 0)
            key_padding_mask = None
        else: 
            memory = torch.cat((memory[-31:], embeddings), 0) 
            # key_padding_mask = torch.cat((key_padding_mask[:,-31:], torch.zeros(2,1,device='cuda')), 1).to(bool)   ## N * S
        out = self.smt_encoder(o = embeddings, M = memory, flag = 0, key_padding_mask = key_padding_mask)

        # T D
        out = torch.cat((out.squeeze(0), task_obs), -1)

        return out, memory


## pytorch N C HW
## tensorflow NHW C
if __name__ == '__main__':
    # encoder = Encoder(feature_dim = 252)
    # rgb = Variable(torch.randn(1, 3, 180, 320))
    # depth = Variable(torch.randn(1, 1, 180, 320))
    # task_obs = Variable(torch.randn(1, 4))
    # print(encoder(rgb,depth,task_obs).size())
    # print(x)

    encoder = Trans_Encoder(feature_dim = 252, state_dim=256)
    rgb = Variable(torch.randn(1, 500, 180, 320, 3))
    depth = Variable(torch.randn(1, 500, 180, 320, 1))
    task_obs = Variable(torch.randn(1, 500, 4))
    print(encoder(rgb,depth,task_obs,1,0)[0].size())
    # print(encoder.self_attn(task_obs,task_obs,task_obs).size())


