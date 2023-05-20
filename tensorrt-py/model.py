import torch
import numpy as np
import tensorrt as trt
from torch import nn as nn
import torch.nn.functional as F
import argparse


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        z = F.softmax(y, dim=1)
        z = torch.argmax(z, dim=1)
        return y, z

def export_onnx(args, model):
    params = torch.load(args.paraFile)
    model.load_state_dict(params['net'])
    torch.onnx.export(
                model,
                torch.randn(1, 1, args.img_sz, args.img_sz, device='cuda'),
                args.onnxFile,
                input_names = ['x'],
                output_names = ['y', 'z'],
                do_constant_folding=True,
                verbose=True,
                keep_initializers_as_inputs=True,
                opset_version=12,
                dynamic_axes={
                    'x':{0: 'inBatchSize'},
                    'z':{0: 'outBatchSize'}
                })
    print('onnx file has exported!')

def config():
    parser = argparse.ArgumentParser(description="test tensorRT")
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--w_d', default=1e-4)
    parser.add_argument('--bs', default=256)
    parser.add_argument("--epochs", default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_w', default=4)
    parser.add_argument('--patience', default=(10, 1e-4))
    parser.add_argument('--img_sz', default=28)
    # parser.add_argument('--paraFile', default='./assets/para.npz')
    parser.add_argument('--paraFile', default='./assets/para.pth')
    parser.add_argument('--onnxFile', default='./assets/model.onnx')
    parser.add_argument('--precision', default=4)
    args = parser.parse_args()
    return args

args = config()
print(args)
model = Net().cuda()
model.train()
model.eval()
params =  {'net':model.state_dict()}
torch.save(params, args.paraFile)
export_onnx(args, model)