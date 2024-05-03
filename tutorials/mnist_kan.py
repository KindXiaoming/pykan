# adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from kan import KAN


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=40, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    model = KAN(width=[28 * 28, 25, 10], grid=5, k=3, seed=0, device=device)
    train_range = [i for i in range(len(dataset1))]
    test_range = [i for i in range(len(dataset2))]
    random.shuffle(train_range)
    dataset_kan = dict(train_input=torch.stack([dataset1[i][0] for i in train_range]).flatten(-3).to(device), 
                       train_label=torch.tensor([dataset1[i][1] for i in train_range]).to(device), 
                       test_input=torch.stack([dataset2[i][0] for i in range(len(dataset2))]).flatten(-3).to(device), 
                       test_label=torch.tensor([dataset2[i][1] for i in range(len(dataset2))]).to(device))

    def train_acc():
        return torch.mean((torch.argmax(model(dataset_kan['train_input'][0:100]), dim=1) == dataset_kan['train_label'][0:100]).float())

    def test_acc():
        return torch.mean((torch.argmax(model(dataset_kan['test_input'][0:100]), dim=1) == dataset_kan['test_label'][0:100]).float())

    result = model.train(dataset_kan, opt="Adam", device=device, batch=args.batch_size, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss(), lr=args.lr)
    print(result)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_kan.pt")


if __name__ == '__main__':
    main()
