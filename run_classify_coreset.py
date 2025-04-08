import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from pytorch_lightning import seed_everything

from data import get_data
from models import get_model
from logger import CSVLogger
from corrupt import add_label_noise
from sampling.sample import run_coreset_select


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--noise_fraction", type=float, default=0)
    parser.add_argument("--log_filename", type=str, default="logs/cifar10/baseline/logs.csv")
    parser.add_argument("--coreset_size_dataset", type=float, default=None)
    parser.add_argument("--coreset_method", type=str, default="random")
    parser.add_argument("--num_repetitions", type=int, default=3)
    return parser.parse_args()


def test(model, loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    model.train()
    return val_acc


def main(args):
    os.makedirs(os.path.dirname(args.log_filename), exist_ok=True)
    csv_logger = CSVLogger(
        fieldnames=['epoch', 'batch', 'train_acc', 'test_acc', 'train_loss'],
        filename=args.log_filename
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # load dataset
    # --------------------------------------------
    tr_dataset, n_classes = get_data(dataset_name=args.dataset, split="train")
    test_dataset, _ = get_data(dataset_name=args.dataset, split="test")

    # Add label noise
    # --------------------------------------------
    if args.noise_fraction > 0:
        print(f"Adding label noise with fraction {args.noise_fraction}")
        if args.noise_fraction > 1:
            raise ValueError("Noise fraction should be between 0 and 1.")
        # Apply noise to the dataset
        tr_dataset = add_label_noise(
            dataset=tr_dataset,
            noise_ratio=args.noise_fraction,
        )

    # Subset Selection
    # --------------------------------------------
    if args.noise_fraction > 0:
        embedding_file_path = os.path.join("embeds", args.dataset + str(args.noise_fraction))
    else:
        # No noise, use the original dataset
        embedding_file_path = os.path.join("embeds", args.dataset)

    os.makedirs(embedding_file_path, exist_ok=True)
    embedding_file = os.path.join(embedding_file_path, "embeddings.pt")
    subset_size = args.coreset_size_dataset

    total_samples = len(tr_dataset)
    print(f"Total samples in dataset: {total_samples}")

    if subset_size and subset_size < total_samples:
        subset_indices = run_coreset_select(
            dataset=tr_dataset,
            coreset_size=subset_size,
            coreset_method=args.coreset_method,
            embedding_file=embedding_file,
            embed_bs=512
        )

        tr_dataset = Subset(tr_dataset, subset_indices)

    else:
        print(f"Using full dataset with {total_samples} samples.")

    # Training
    # --------------------------------------------
    if args.dataset == "cifar10":
        # We will run ResNet18 on CIFAR10 --
        # Hyperparameters are taken from the paper:

        # Improved Regularization of Convolutional Neural Networks with Cutout
        # http://arxiv.org/abs/1708.04552
        # github: https://github.com/uoguelph-mlrg/Cutout

        model = get_model(model_name="CIFARResNet18", num_classes=10)

        optimizer = SGD(
            model.parameters(), 
            lr=0.1, 
            momentum=0.9, 
            nesterov=True,
            weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(
            optimizer,
            milestones=[60, 120, 160],
            gamma=0.2
        )
        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size=128,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
        num_epochs = 200

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    model = model.to(device)
    best_acc = 0.

    for epoch in range(num_epochs):
        acc = 0.
        loss_avg = 0.
        correct = 0.
        total = 0.

        pbar = tqdm(tr_dataloader)

        for batch_idx, (images, labels) in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_avg += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = correct / total
            current_loss = loss_avg / (batch_idx + 1)
            pbar.set_postfix(
                loss='%.3f' % current_loss,
                acc='%.3f' % acc
            )
            
            # Log every batch
            row = {
                'epoch': str(epoch),
                'batch': str(batch_idx),
                'train_acc': str(acc),
                'test_acc': '',  # Empty for batch updates
                'train_loss': str(current_loss)
            }
            csv_logger.writerow(row)

        scheduler.step()
        test_acc = test(model, test_dataloader)
        if test_acc > best_acc:
            best_acc = test_acc
            print("Best accuracy so far: %.3f" % best_acc)

        tqdm.write('test_acc: %.3f' % test_acc)
        row = {
            'epoch': str(epoch),
            'batch': 'end',
            'train_acc': str(acc),
            'test_acc': str(test_acc),
            'train_loss': str(loss_avg / len(tr_dataloader))
        }
        csv_logger.writerow(row)

    print("Finished Training")
    csv_logger.close()
    return best_acc


if __name__ == "__main__":
    args = parse_args()
    print(args)
    n_repetitions = args.num_repetitions
    best_acc = []
    for i in range(n_repetitions):
        print(f"Repetition {i + 1}/{n_repetitions}")
        seed_everything(i)
        acc = main(args)
        print(f"Seed/Best: {i}/{acc}")
        best_acc.append(acc)

    print("All repetitions finished.")
    print(f"Best accuracy so far: {best_acc}")
    print(f"Best accuracy over {n_repetitions} repetitions: {max(best_acc)}")
    mean_acc = 100 * np.mean(best_acc)
    std = 100 * np.std(best_acc)

    print(f"Mean accuracy: {mean_acc:.3f} ± {std:.3f}")



