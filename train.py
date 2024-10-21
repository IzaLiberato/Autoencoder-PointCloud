import torch
from pytorch3d.loss import chamfer_distance
import matplotlib.pyplot as plt
import utils

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    epoch_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data.permute(0, 2, 1))
        loss, _ = chamfer_distance(data, output)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def test_epoch(model, test_loader, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.permute(0, 2, 1))
            loss, _ = chamfer_distance(data, output)
            epoch_loss += loss.item()
    return epoch_loss / len(test_loader)

def save_results(train_loss_list, test_loss_list, model, test_loader, epoch, output_folder):
    # Plot and save loss graph
    plt.plot(train_loss_list, label="Train")
    plt.plot(test_loss_list, label="Test")
    plt.legend()
    plt.savefig(f"{output_folder}/loss_epoch_{epoch}.png")
    plt.close()

    # Save input/output samples
    test_samples = next(iter(test_loader))
    loss, test_output = test_epoch(model, test_loader, model.device)
    utils.plotPCbatch(test_samples, test_output, save=True, name=f"{output_folder}/samples_epoch_{epoch}")
