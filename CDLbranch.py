
import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,
                 dim):
        super(Attention, self).__init__()
        self.scale= dim ** -0.5
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)


    def forward(self,  x,dictionary,S):
        B1, N1, C1=x.shape
        B2, N2, C2=dictionary.shape
        S=S.unsqueeze(2)
        dictionary=dictionary*S
        q = self.fc1(x).reshape(B1, N1, 1,C1).permute(2, 0,  1, 3)
        k = self.fc2(dictionary).reshape(B2, N2, 1,C2).permute(2, 0,  1, 3)
        q, k = q[0], k[0]
        T = (q @ k.transpose(-2, -1)) * self.scale

        tavg=torch.mean(T,dim=2,keepdim=True)

        tatt = tavg.softmax(dim=1)
        tatt=tatt.repeat(1,1,C1)
        x=x*tatt
        return x

class GSAM(nn.Module):
    def __init__(self,dim,norm_layer=nn.LayerNorm):
        super(GSAM, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim)

    def forward(self, x,dictionary,S):
        x = x + self.attn(self.norm1(x),dictionary,S).squeeze(1)
        return x


class SSM(nn.Module):
    def __init__(self,args):
        super(SSM, self).__init__()
        self.fc_d1 = nn.Linear(2048,2048)
        self.fc_f = nn.Linear(2048,args.num_classes)
        self.lam=1
    def forward(self ,dictionary,tf):
        fcd=self.fc_d1(dictionary)
        p1=self.fc_f(tf)
        Q_x = torch.mm(fcd, torch.transpose(fcd, 0, 1))
        b_x = torch.mm(tf.detach(), torch.transpose(fcd, 0, 1))
        I_x = torch.eye(Q_x.shape[0]).cuda()
        G_x = Q_x + self.lam * I_x
        G_x = torch.inverse(G_x)
        S_x = torch.mm(G_x, torch.transpose(b_x, 0, 1))
        S_x = torch.transpose(S_x, 0, 1 )
        d=torch.mm(S_x,fcd)
        return S_x,p1,d

class GroupingUnit(nn.Module):
    def __init__(self,args, dim, num_classes):
        super(GroupingUnit, self).__init__()
        self.num_parts = num_classes
        self.num_classes = num_classes
        self.dim = dim
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, dim))
        self.fcp2 = nn.Sequential(
            nn.Linear(dim, args.num_classes)
        )
        self.GSAM1= GSAM(dim=dim)
        self.GSAM2= GSAM(dim=dim)
        self.GSAM3= GSAM(dim=dim)

        self.SSM=SSM(args)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.CM=nn.Sequential(
            nn.Conv2d(self.dim,self.dim,3,1,1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
            nn.Conv2d(self.dim,self.dim,1,1),
            nn.BatchNorm2d(self.dim),
        )
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        self.weight.data.clamp_(min=1e-5)


    def forward(self, x,features=None):
        batch_size = features[0].size(0)
        dictionary=self.weight.contiguous().view(self.num_parts, self.dim)
        S_x,p1,d=self.SSM(dictionary,x)
        H3=self.GSAM1(features[0],dictionary.unsqueeze(0).repeat(batch_size,1,1),S_x)
        H4=self.GSAM2(features[1],dictionary.unsqueeze(0).repeat(batch_size,1,1),S_x)
        H5=self.GSAM3(features[2],dictionary.unsqueeze(0).repeat(batch_size,1,1),S_x)

        H, W=int(math.sqrt(H3.shape[1])),int(math.sqrt(H3.shape[1]))
        H3 = H3.permute(0,2,1).reshape(batch_size,-1,H,W)
        H4 = H4.permute(0,2,1).reshape(batch_size,-1,H,W)
        H5 = H5.permute(0,2,1).reshape(batch_size,-1,H,W)
        edge = self.CM(H3+H4+H5)
        edge = self.avg_pool(edge).flatten(1)
        p2 = self.fcp2(edge)

        return p1,d,p2,[H3,H4,H5]

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.num_parts) + ')'


class Branch(nn.Module):
    def __init__(self, args=None):
        super(Branch, self).__init__()
        self.dim=2048
        self.conv3=nn.Sequential(
            nn.Conv2d(self.dim//2//2,self.dim,4,4),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(self.dim//2,self.dim,2,2),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(self.dim,self.dim,1,1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.grouping = GroupingUnit(args,self.dim,args.num_classes)
        self.grouping.reset_parameters()
        self.apply(_init_vit_weights)

    def forward(self ,featuremap):
        E3, E4, E5=featuremap
        x = self.avg_pool(E5).flatten(1)
        F3=self.conv3(E3)
        F4=self.conv4(E4)
        F5=self.conv5(E5)
        F3=F3.permute(0, 2, 3, 1).reshape(F3.shape[0], F3.shape[2] * F3.shape[3], -1)
        F4=F4.permute(0, 2, 3, 1).reshape(F4.shape[0], F4.shape[2] * F4.shape[3], -1)
        F5=F5.permute(0, 2, 3, 1).reshape(F5.shape[0], F5.shape[2] * F5.shape[3], -1)
        p1, d, p2, att = self.grouping(x,[F3,F4,F5])
        return x, p1, d, p2, att



def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()