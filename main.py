import torch
import time
import matplotlib.pyplot as plt
from config import config
from data import GetDataLoaders
from model import PointCloudAE
from train import train_epoch, test_epoch, save_results
import utils

# Configurações
output_folder = config['output_folder']
use_GPU = config['use_GPU']
latent_size = config['latent_size']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
save_results = config['save_results']

# Carregar dados
pc_array = np.load(config['data_path'])
train_loader, test_loader = GetDataLoaders(pc_array, batch_size)

# Definir o modelo e dispositivo (GPU/CPU)
point_size = len(train_loader.dataset[0])
net = PointCloudAE(point_size, latent_size)
device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")
net = net.to(device)

# Otimizador
optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])

# Treinamento
train_loss_list, test_loss_list = [], []
for epoch in range(num_epochs):
    start_time = time.time()
    
    train_loss = train_epoch(net, train_loader, optimizer, device)
    test_loss = test_epoch(net, test_loader, device)
    
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Time = {epoch_time:.2f}s")
    
    # Salvar resultados
    if save_results and epoch % 50 == 0:
        save_results(train_loss_list, test_loss_list, net, test_loader, epoch, output_folder)

