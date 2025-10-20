import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data_utils
from torch.utils.data import DataLoader
from causal_discover import CausalityGraph4ML1M

class CausalMLP(nn.Module):
    def __init__(self, embedding_dim=64):
        super(CausalMLP, self).__init__()
        self.input_dim = 54
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim * 2)  
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        output = self.mlp(x)
        user_embedding, item_embedding = torch.split(output, self.embedding_dim, dim=1)
        return user_embedding, item_embedding

def main():
    user_num = 6040
    item_num = 3952
    batch_size = 2048 * 100
    
    train_file = "../data/ml/train.npy"
    users_features_file = "../data/ml/users_features.npy"
    movies_features_file = "../data/ml/movie_features.npy"
    
    train_dict, train_dict_count = np.load(train_file, allow_pickle=True)
    
    train_dataset = data_utils.BPRData(
        train_dict=train_dict,
        num_item=item_num,
        num_ng=5,
        is_training=0,
        data_set_count=train_dict_count
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    print(f"lendata: {len(train_dataset)}")
    print(f"num batch: {len(train_loader)}")
    
    causal_graph = CausalityGraph4ML1M(train_file, users_features_file, movies_features_file)
    model = CausalMLP(embedding_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    user_embeddings_dict = {}  
    item_embeddings_dict = {}  
    
    model.eval()  
    train_loader.dataset.ng_sample()  
    
    with torch.no_grad():
        for user_batch, item_i_batch, _ in train_loader:  
            causal_matrices = []
            for user_id, item_id in zip(user_batch, item_i_batch):
                causal_matrix = causal_graph.get_user_item_causal_matrix(
                    user_id.item(), 
                    item_id.item()
                )
                causal_matrices.append(causal_matrix)
            
            batch_matrices = np.array(causal_matrices)
            batch_matrices = torch.FloatTensor(batch_matrices).to(device)
            user_emb, item_emb = model(batch_matrices)
            
            for i in range(len(user_batch)):
                user_id = user_batch[i].item()
                item_id = item_i_batch[i].item()
                user_embeddings_dict[user_id] = user_emb[i].cpu()
                item_embeddings_dict[item_id] = item_emb[i].cpu()
    
    torch.save({
        'user_embeddings_dict': user_embeddings_dict,
        'item_embeddings_dict': item_embeddings_dict
    }, './causal_embeddings.pt')
    user_emb_tensor = torch.stack(list(user_embeddings_dict.values()))
    # print("\nUser embeddings statistics:")
    # print(f"Max: {user_emb_tensor.max().item():.4f}")
    # print(f"Min: {user_emb_tensor.min().item():.4f}")
    # print(f"Mean: {user_emb_tensor.mean().item():.4f}")

    item_emb_tensor = torch.stack(list(item_embeddings_dict.values()))
    print("\nItem embeddings statistics:")
    print(f"Max: {item_emb_tensor.max().item():.4f}")
    print(f"Min: {item_emb_tensor.min().item():.4f}")
    print(f"Mean: {item_emb_tensor.mean().item():.4f}")
    print("succeed,save causal_embeddings.pt")
    # print(f"num user emb: {len(user_embeddings_dict)}")
    # print(f"num item emb: {len(item_embeddings_dict)}")

if __name__ == "__main__":
    main()
