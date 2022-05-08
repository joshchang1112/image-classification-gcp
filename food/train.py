import os
import argparse
import time
import torch
import torchvision
from tqdm import tqdm
from metrics import Accuracy
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from utils import set_seed, load_data

def training(args, train_loader, valid_loader, model, optimizer, scheduler, device):
    train_metrics = Accuracy()
    best_valid_acc = 0
    criterion = torch.nn.CrossEntropyLoss()
    total_time = 0
    early_stopping = 0
    for epoch in range(args.epochs):
        start = time.time()
        train_trange = tqdm(enumerate(train_loader), total=len(train_loader), desc='training')
        train_loss = 0
        train_metrics.reset()
        for i, (feat, answer) in train_trange:
            model.train()
            feat, answer = feat.to(device), answer.to(device)
            prob = model(feat)
            loss = criterion(prob, answer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_metrics.update(prob, answer)
            train_trange.set_postfix(loss= train_loss/(i+1),
                                     **{train_metrics.name: train_metrics.print_score()})

        train_acc = train_metrics.match / train_metrics.n
        valid_acc = testing(valid_loader, model, device, valid=True)
        scheduler.step(valid_acc)
        if valid_acc > best_valid_acc:
            early_stopping = 0
            best_valid_acc = valid_acc
            torch.save(model, os.path.join(args.model_dir, 'best_resnet50.pkl'))
            #torch.save(model, os.path.join(args.model_dir, 'best_mobile_net_v3.pkl'))
            print('Best Valid Accuracy: {:.4f}'.format(best_valid_acc))
        early_stopping += 1
        end = time.time()
        print("Epoch {}, Train Acc: {:.4f}, Validation Acc: {:.4f}, Time: {:.4f}".format(
            epoch +1, train_acc, valid_acc, end-start))
        total_time += (end-start)
        if early_stopping == 10:
            print("EARLY STOPPING at Epoch {}".format(str(epoch+1)))
            break
    print("Total Time: {}".format(total_time))

def testing(dataloader, model, device, valid):
    metrics = Accuracy()
    criterion = torch.nn.CrossEntropyLoss()
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='validation' if valid else 'testing')
    model.eval()
    total_loss = 0
    metrics.reset()
    for k, (feat, answer) in trange:
        model.eval()
        feat, answer = feat.to(device), answer.to(device)
        prob = model(feat) 
        loss = criterion(prob, answer)

        total_loss += loss.item()
        metrics.update(prob, answer)
        trange.set_postfix(loss= total_loss/(k+1),
                           **{metrics.name: metrics.print_score()})
    acc = metrics.match / metrics.n
    return acc

def main():
    parser = argparse.ArgumentParser(description='Image Classification for Food')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_dir', default='models', type=str,
                        help='Directory to the model checkpoint.')
    
    args = parser.parse_args()
    if os.path.isdir(args.model_dir) == False:
        os.makedirs(args.model_dir)
        print('Create folder: {}'.format(args.model_dir))
    else:
        print('{} exists!'.format(args.model_dir))

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available()
                          else 'cpu')
    
    set_seed(SEED=args.seed)
    train_dataset, valid_dataset, test_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 101)
    #model = models.mobilenet_v3_small(pretrained=True)
    #model.classifier[-1] = torch.nn.Linear(1024, 101)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    training(args, train_loader, valid_loader, model, optimizer, scheduler, device) 
    
    #model = torch.load(os.path.join(args.model_dir, 'best_mobile_net_v3.pkl'))
    model = torch.load(os.path.join(args.model_dir, 'best_resnet50.pkl'))
    print(testing(test_loader, model, device, valid=False))

if __name__ == "__main__":
    main()
