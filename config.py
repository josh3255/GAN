import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('GAN Parameters', add_help=False)

    # train and valid params
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--epochs', default=6000, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument("--batch_size", type=int, default=512, help="train batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")

    parser.add_argument('--train_ratio', default=2, type=int)
    parser.add_argument('--save_term', default=50, type=int)

    parser.add_argument('--train_root_path', default='/data/mnist')
    parser.add_argument('--train_img_size', default=(1, 28, 28), nargs='+',type=int)

    # test params
    parser.add_argument('--test_term', default=5, type=int)
    parser.add_argument("--test_batch_size", type=int, default=10, help="test batch size")
    parser.add_argument('--test_ckp', default='./weights/generator_50.pth')

    parser.add_argument("--latent_dimension", type=int, default=100, help="dimensionality of the latent space")
    return parser