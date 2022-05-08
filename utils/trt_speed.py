import torch
import torchvision
from torch2trt import torch2trt


output_dim = 2
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, output_dim)
model = model.cuda().eval().half()

basename_list = ["../0422school_A/choicest_batch16_epoch15_RESNET_18",
                 "../0422school_A/choicest_batch32_epoch15_RESNET_18",
                 "../0422school_A/choicest_batch48_epoch15_RESNET_18",
                 "../0422school_A/choicest_batch64_epoch15_RESNET_18",
                 ]

for basename in basename_list:
    model.load_state_dict(torch.load('{}.pth'.format(basename)))
    data = torch.zeros((1, 3, 224, 224)).cuda().half()
    model_trt = torch2trt(model, [data], fp16_mode=True)
    torch.save(model_trt.state_dict(), '{}_trt.pth'.format(basename))
    print("{}_trt.pth".format(basename))
