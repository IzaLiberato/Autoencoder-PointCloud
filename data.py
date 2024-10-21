import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ReadDataset(Dataset):
    def __init__(self, source):
        self.data = torch.from_numpy(source).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets) * train_set_percentage), len(datasets) - int(len(datasets) * train_set_percentage)]
    return random_split(datasets, lengths)

def GetDataLoaders(npArray, batch_size, train_set_percentage=0.9):
    dataset = ReadDataset(npArray)
    train_set, test_set = RandomSplit(dataset, train_set_percentage)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
