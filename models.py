import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MyNet(nn.Module):
    def __init__(self, code_len, ori_featI, ori_featT):
        super(MyNet, self).__init__()
        self.code_len = code_len

        ''' IRR_img '''
        self.encoderIMG = nn.Sequential(
            nn.Linear(ori_featI, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.gcnI1 = nn.Linear(512, 512)
        self.BNI1 = nn.BatchNorm1d(512)
        self.actI1 = nn.ReLU(inplace=True)

        self.decoderIMG = nn.Sequential(
            nn.Linear(code_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, ori_featI),
            nn.BatchNorm1d(ori_featI)
            )

        '''IRR_txt'''
        self.encoderTXT = nn.Sequential(
            nn.Linear(ori_featT, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.gcnT1 = nn.Linear(512, 512)
        self.BNT1 = nn.BatchNorm1d(512)
        self.actT1 = nn.ReLU(inplace=True)

        self.decoderTXT = nn.Sequential(
            nn.Linear(code_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, ori_featT),
            nn.BatchNorm1d(ori_featT)
            )

        '''CMA'''
        self.gcnJ1 = nn.Linear(512, 512)
        self.BNJ1 = nn.BatchNorm1d(512)
        self.actJ1 = nn.ReLU(inplace=True)

        self.HJ = nn.Linear(512, code_len)

        self.img_fc = nn.Linear(2 * 512, code_len)
        self.txt_fc = nn.Linear(2 * 512, code_len)
        self.HC = nn.Linear(3 * code_len, code_len)

        self.HIBN = nn.BatchNorm1d(code_len)
        self.HTBN = nn.BatchNorm1d(code_len)
        self.HJBN = nn.BatchNorm1d(code_len)
        self.HBN = nn.BatchNorm1d(code_len)

    def forward(self, XI, XT, affinity_A):

        self.batch_num = XI.size(0)

        ''' IRR_img '''
        VI = self.encoderIMG(XI)
        VI = F.normalize(VI, dim=1)

        VgcnI = self.gcnI1(VI)
        VgcnI = affinity_A.mm(VgcnI)
        VgcnI = self.BNI1(VgcnI)
        VgcnI = self.actI1(VgcnI)

        ''' IRR_txt '''
        VT = self.encoderTXT(XT)
        VT = F.normalize(VT, dim=1)

        VgcnT = self.gcnT1(VT)
        VgcnT = affinity_A.mm(VgcnT)
        VgcnT = self.BNT1(VgcnT)
        VgcnT = self.actT1(VgcnT)

        '''CMA'''
        VC = torch.cat((VI, VT), 0)
        II = torch.eye(affinity_A.shape[0], affinity_A.shape[1]).cuda()
        S_cma = torch.cat((torch.cat((affinity_A, II), 1),
                            torch.cat((II, affinity_A), 1)), 0)

        VJ1 = self.gcnJ1(VC)
        VJ1 = S_cma.mm(VJ1)
        VJ1 = self.BNJ1(VJ1)
        VJ1 = VJ1[:self.batch_num, :] + VJ1[self.batch_num:, :]
        VJ = self.actJ1(VJ1)

        HJ = self.HJ(VJ)
        HJ = self.HJBN(HJ)

        HI = self.HIBN(self.img_fc(torch.cat((VgcnI, VJ), 1)))
        HT = self.HTBN(self.txt_fc(torch.cat((VJ, VgcnT), 1)))

        H = torch.tanh(self.HBN(self.HC(torch.cat((HI, HJ, HT), 1))))

        B = torch.sign(H)

        DeI_feat = self.decoderIMG(H + torch.tanh(HI))
        DeT_feat = self.decoderTXT(H + torch.tanh(HT))

        return HI, HT, H, B, DeI_feat, DeT_feat


class Img_Net(nn.Module):
    def __init__(self, code_len, img_feat_len):
        super(Img_Net, self).__init__()

        self.fc1 = nn.Linear(img_feat_len, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HI = self.tanh(hid)

        return HI

class Txt_Net(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(Txt_Net, self).__init__()

        self.fc1 = nn.Linear(txt_feat_len, txt_feat_len)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(txt_feat_len, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HT = self.tanh(hid)

        return HT