"""
Author: Pooja Choudhary
Student ID: 48449496

"""


import torch
from modules import VQVAE
from dataset import get_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "C:\\Users\\pooja\\Documents\\PatternAnalysis-2025\\recognition\\VQVAE_s4844949\\Data\\keras_slices_data\\"
    train_loader, _, _ = get_dataloaders(root, batch_size=8, target_size=(256,128))

    model = VQVAE(embedding_dim=64, num_embeddings=512, commitment_cost=0.25, ema_decay=0.99).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)


    iters = 50
    loader_iter = iter(train_loader)
    for t in range(1, iters + 1):
        #TODO write the logic later
        pass

if __name__ == "__main__":
    main()