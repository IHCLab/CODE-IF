import torch
import torch.nn as nn
import numpy as np, math
from torch.nn import functional as F
from torch.autograd import Variable
#import matplotlib.pyplot as plt
    
##===================================

##===================================
# Attention block (Traditional)
class ATB(nn.Module):
    def __init__(self, nMSband):
        super().__init__()
        
        self.k3_10=nn.Sequential(
        nn.Conv2d(in_channels=nMSband, out_channels=10, kernel_size=3, stride=1, padding=1),
        nn.PReLU()
        )
        self.k5_10=nn.Sequential(
        nn.Conv2d(in_channels=nMSband, out_channels=10, kernel_size=5, stride=1, padding=2),
        nn.PReLU()
        )
        
        self.k7_10=nn.Sequential(
        nn.Conv2d(in_channels=nMSband, out_channels=10, kernel_size=7, stride=1, padding=3),
        nn.PReLU()
        )
        
        self.ConvCatt=nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )
        
        self.ConvSatt=nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5 ,stride=1 ,padding=2),
            nn.PReLU(),
        )
        
        self.attR=nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=15, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=15, out_channels=nMSband, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        
        self.k3_mid=nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )

    def forward(self, inp):
        
        ym3=self.k3_10(inp)
        ym5=self.k5_10(inp)
        ym7=self.k7_10(inp)
        yc=torch.cat((ym3,ym5,ym7),1)
       
        
        C_GAPa=torch.mean(yc, (2,3),keepdim=True)
        Catt=self.ConvCatt(C_GAPa)
        Catt=yc*Catt
        
        Catt=self.k3_mid(Catt)
        
        S_GAPa=torch.mean(Catt, 1,keepdim=True)
        S_GAPm,_=torch.max(Catt, 1,keepdim=True)
        Satt=self.ConvSatt(torch.cat((S_GAPa,S_GAPm),1))
        Satt=Satt*Catt
        
        ATT=self.attR(Satt)
        x=inp+ATT
        
        return x

##===================================              
class FusionNet(nn.Module): 
    def __init__(self ,ratio, nMSband, nHSband):
        super(FusionNet, self).__init__()
                         
        self.k3_1=nn.Sequential(
        nn.Conv2d(in_channels=nHSband+nMSband*6, out_channels=256, kernel_size=3, stride=1, padding=1), # 256
        nn.PReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), #256 256
        nn.PReLU(),
        nn.Conv2d(in_channels=256, out_channels=nHSband, kernel_size=3, stride=1, padding=1), #256 
        nn.PReLU(),
        )
                  
        self.up = nn.Upsample(mode='bilinear',scale_factor=ratio)
        
        #self.mean = nn.Upsample(mode='bilinear',scale_factor=1/ratio) 
        self.mean = nn.AvgPool2d(kernel_size=ratio, stride=ratio) 
        
        self.atb1=ATB(nMSband)
        self.atb2=ATB(nMSband)
        self.atb3=ATB(nMSband)
        self.atb4=ATB(nMSband)
        self.atb5=ATB(nMSband)

    def forward(self,ym,yh):

        yhup=self.up(yh)
        
        yc1=self.atb1(ym)
        yc2=self.atb2(yc1)
        yc3=self.atb3(yc2)
        yc4=self.atb4(yc3)
        yc5=self.atb5(yc4)

        yc=torch.cat((yhup,yc1,yc2,yc3,yc4,yc5,ym),1)


        yc=self.k3_1(yc)
        
        # zero mean normalize process
        zmyc=self.mean(yc)
        zmyc=self.up(zmyc)
        yc=yc-zmyc
        
        yc=yc+yhup
        
        return yc

#====================================
    

class myModel(nn.Module): 
    def __init__(self ,ratio, nMSband, nHSband):
        super(myModel, self).__init__()
        
        self.fm=FusionNet(ratio, nMSband, nHSband)
        
        self.k=nn.Sequential(
        nn.Conv2d(in_channels=nHSband, out_channels=nHSband, kernel_size=3, stride=1,padding=1),
        nn.PReLU(),
        )
        
    def forward(self,ym,yh):
        
        # FusionNet
        a= self.fm(ym,yh)

        e= self.k(a)
        
        return e
    
##==========================================