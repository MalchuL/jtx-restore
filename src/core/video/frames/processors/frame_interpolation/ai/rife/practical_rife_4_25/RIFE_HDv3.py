import torch
from .IFNet_HDv3 import IFNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.version = 4.25

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):            
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                # Remove teacher and caltime parameters
                if "module." in k and "teacher." not in k and "caltime." not in k
            }
        # I enabled strict=True to ensure that the model is loaded correctly to avoid missing keys.
        if torch.cuda.is_available():
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), strict=True)
        else:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location ='cpu')), strict=True)
        
    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]
