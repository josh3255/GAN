from config import *
from models import *

import numpy as np
import cv2
import os
from torchvision.utils import save_image

import torch.nn
from torch.autograd import Variable

def test(args):
    generator = Generator(args)
    
    if args.test_ckp is not '':
        checkpoint = torch.load(args.test_ckp)
        generator.load_state_dict(checkpoint['model_state_dict'])    

    Tensor = torch.FloatTensor
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
        generator = generator.cuda()

    generator.eval()
    latent_vector = Variable(Tensor(np.random.normal(0, 1, (args.test_batch_size, args.latent_dimension))))
    print(latent_vector)
    generated_img = generator(latent_vector)
    
    for i in range(0, args.test_batch_size):
        save_img = generated_img[i]
        save_image(save_img.data[:25], "./test_output/" + str(i) + ".png", nrow=5, normalize=True)
        
def main(args):
    test(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameters parser', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)