from trainer import DGNet_Trainer
#from utils.utils import get_classes
from utils import get_config
from torchstat import stat
import torch
from torchsummary import summary
from thop import profile


# classes_path = 'model_data/underwater_classes.txt'
# # classes_path = 'model_data/voc_classes.txt'
#
# # ----------------------------------------------------#
# #   获取classes和anchor
# # ----------------------------------------------------#
# class_names, num_classes = get_classes(classes_path)

# ------------------------------------------------------#
#   所使用的YoloX的版本。nano、tiny、s、m、l、x
# ------------------------------------------------------#
# phi = 'm'

#################################################################################################################
# module可选：
# 无
# SEM, ECAM, CBAM, CBAM(S), CBAM(C), BAM, BAM(S), BAM(C), CoAM,ShAM_S
# PSAM, GSoPM, GSoPM(C), GSoPM(S), SGEM, SRM, GCT, RGAM, RGAM(S), RGAM(C), FCAM, SCCM, DAM, TAM, ShAM, SplAM
# HAM, LGLM
# SX4,NL
#module = "SX4"
#################################################################################################################
#model = YoloBody(num_classes, phi, module)  # ！！！！！！！！！！！！！！！！！！！！！！！
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)
config = get_config('configs/latest.yaml')
model = DGNet_Trainer(config, '0').to(device)
input = torch.zeros((1,3,256,128)).to(device)
flops, params = profile(model, inputs=(input,input,input,input))
print('参数:', params)
print('计算量:', flops/1000000000.0, 'GFLOPS')
#stat(model, (3, 256, 128))  # CPU统计
# summary(model.cuda(),input_size=(3,640,640),batch_size=1)  # GPU统计

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))




