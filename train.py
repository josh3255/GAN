from models import *
from config import *

import statistics

import random

from tqdm import tqdm
from tqdm import trange

import torch.nn
import torch.optim as optim

from torchvision.utils import save_image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def train(args):
    
    # fix seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    

    generator = Generator(args)
    discriminator = Discriminator(args)
    adversarial_loss = torch.nn.BCELoss()

    Tensor = torch.FloatTensor
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        adversarial_loss = adversarial_loss.cuda()
    
    download_root = './MNIST_DATASET'
    mnist_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])
    train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
    valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
    test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=15, gamma=0.1)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=15, gamma=0.1)

    print('train data length : {}'.format(len(train_dataloader.dataset)))
    print('valid data length : {}'.format(len(valid_dataloader.dataset)))
    print('test data length : {}'.format(len(test_dataloader.dataset)))
    try:
        for epoch in range(0, args.epochs):
            g_train_epoch_losses = []
            d_train_epoch_losses = []
            
            generator.train()
            discriminator.train()
            
            train_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            # test_pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            
            for iter, data in train_pbar:
                imgs, labels = data
                # print(imgs.shape, labels.shape)
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                # print(valid.shape, fake.shape)
            
                real_imgs = Variable(imgs.type(Tensor))
                
                # train generator
                for i in range(0, args.train_ratio * 3):
                    g_optimizer.zero_grad()
                    z = Variable(Tensor(np.random.normal(0, 1, (imgs.size(0), args.latent_dimension))))
                    gen_imgs = generator(z)
                    g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                    g_train_epoch_losses.append(g_loss.item())
                    g_loss.backward()
                    g_optimizer.step()

                # train discriminator
                for i in range(0, args.train_ratio):
                    d_optimizer.zero_grad()
                    real_loss = adversarial_loss(discriminator(real_imgs), valid)
                    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                    d_train_epoch_losses.append(d_loss.item())
                    d_loss.backward()
                    d_optimizer.step()
        
                train_pbar.set_description('epoch : {} || g_lr : {} || d_lr : {} || g_loss : {} || d_loss : {}'.format(epoch, g_optimizer.param_groups[0]['lr'], d_optimizer.param_groups[0]['lr'], statistics.mean(g_train_epoch_losses), statistics.mean(d_train_epoch_losses)))

            g_scheduler.step()
            d_scheduler.step()

            if epoch != 0 and epoch % args.test_term == 0:
                generator.eval()
                latent_vector = Variable(Tensor(np.random.normal(0, 1, (args.test_batch_size, args.latent_dimension))))
                g_outputs = generator(latent_vector)
                for i in range(args.test_batch_size):
                    save_image(g_outputs[i].data[:25], "./test_output/epoch_" + str(epoch) + '_' + str(i) + ".png", nrow=5, normalize=True)
            
            torch.save({
                'epoch' : epoch,
                'model_state_dict' : generator.state_dict(),
                'optimizer_state_dict' : g_optimizer.state_dict(),
                'scheduler_state_dict' : g_scheduler.state_dict(),
            }, 'weights/generator_latest.pth')
            
            if epoch != 0 and epoch % args.save_term == 0:
                torch.save({
                    'epoch' : epoch,
                    'model_state_dict' : generator.state_dict(),
                    'optimizer_state_dict' : g_optimizer.state_dict(),
                    'scheduler_state_dict' : g_scheduler.state_dict(),
                }, 'weights/generator_' + str(epoch) + '.pth')

            # print('epoch : {} || g_lr : {} || d_lr : {} || g_loss : {} || d_loss : {}'.format(epoch, g_optimizer.param_groups[0]['lr'], d_optimizer.param_groups[0]['lr'], statistics.mean(g_train_epoch_losses), statistics.mean(d_train_epoch_losses)))
    except Exception as e:
        print(e)
        print(real_imgs.shape, valid.shape)
            
def main(args):
    train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameters parser', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)