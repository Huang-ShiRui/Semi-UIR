import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
from adamp import AdamP
# my import
from model import AIMnet
from dataset_all import TestData

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

bz = 1
model_root = 'pretrained/model.pth'
input_root = 'data/test/benchmarkA/'
save_path = 'result/benchmarkA'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
checkpoint = torch.load(model_root)
Mydata_ = TestData(input_root)
data_load = data.DataLoader(Mydata_, batch_size=bz)

model = AIMnet().cuda()
model = nn.DataParallel(model, device_ids=[0, 1])
optimizer = AdamP(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_dict'])
epoch = checkpoint['epoch']
model.eval()
print('START!')
if 1:
    print('Load model successfully!')
    for data_idx, data_ in enumerate(data_load):
        data_input, data_la = data_

        data_input = Variable(data_input).cuda()
        data_la = Variable(data_la).cuda()
        print(data_idx)
        with torch.no_grad():
            result, _ = model(data_input, data_la)
            name = Mydata_.A_paths[data_idx].split('/')[5]
            print(name)
            temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
            temp_res[temp_res > 1] = 1
            temp_res[temp_res < 0] = 0
            temp_res = (temp_res*255).astype(np.uint8)
            temp_res = Image.fromarray(temp_res)
            temp_res.save('%s/%s' % (save_path, name))
            print('result saved!')

print('finished!')
