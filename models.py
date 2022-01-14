import cv2

import torch
import numpy as np
import torch.nn as nn

# from config import *

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__() 
        self.args = args
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.args.train_img_size)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        def block(in_feature, out_feature, normalize=True):
            layers = [nn.Linear(in_feature, out_feature)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feature, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(args.latent_dimension, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.args.train_img_size)))
        )

    def forward(self, z):
        generated_image = self.model(z)
        generated_image = generated_image.view(generated_image.size(0), *self.args.train_img_size)
        
        return generated_image

def main(args):
    g_model = Generator(args)
    tmp = torch.randn((4, args.latent_dimension))
    # tmp = torch.randn(args.train_img_size)
    g_output = g_model(tmp)
    
    # g_output = g_output.detach().numpy()
    # save_g_output = g_output[0].transpose((1, 2, 0))
    # print(save_g_output)
    # cv2.imwrite('./test.jpg', save_g_output)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameters parser', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)


