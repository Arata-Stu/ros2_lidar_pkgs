import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SplineConv, global_mean_pool, global_max_pool

class ModularLidarGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, pool_method='mean'):
        super(ModularLidarGCN, self).__init__()

        # --- GCN Layers ---
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim // (2 ** i), hidden_dim // (2 ** (i + 1))))

        # --- Poolingの選択 ---
        self.pool_method = pool_method
        
        # --- MLPによる予測 ---
        final_dim = hidden_dim // (2 ** (num_layers - 1))
        self.fc1 = torch.nn.Linear(final_dim, final_dim // 2)
        self.fc2 = torch.nn.Linear(final_dim // 2, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # --- Residual Connections 処理 ---
        residuals = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = F.relu(layer(x, edge_index))
            else:
                # 前の層の出力を足し込む（Residual）
                x = F.relu(layer(x + residuals[-1], edge_index))
            residuals.append(x)

        # --- プーリング処理 (Mean or Max) ---
        if self.pool_method == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_method == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError("Invalid pool method")
        
        # --- MLPによる予測 ---
        x = F.relu(self.fc1(x))
        action = self.fc2(x)  # 出力: [steer, speed]
        
        return action

class ModularLidarSplineGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, pool_method='mean', kernel_size=3):
        super(ModularLidarSplineGCN, self).__init__()

        # --- SplineConv Layers ---
        self.layers = torch.nn.ModuleList()
        self.layers.append(SplineConv(input_dim, hidden_dim, dim=2, kernel_size=kernel_size))
        for i in range(num_layers - 1):
            self.layers.append(SplineConv(hidden_dim // (2 ** i), hidden_dim // (2 ** (i + 1)), dim=2, kernel_size=kernel_size))

        # --- Poolingの選択 ---
        self.pool_method = pool_method
        
        # --- MLPによる予測 ---
        final_dim = hidden_dim // (2 ** (num_layers - 1))
        self.fc1 = torch.nn.Linear(final_dim, final_dim // 2)
        self.fc2 = torch.nn.Linear(final_dim // 2, output_dim)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # --- Residual Connections 処理 ---
        residuals = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = F.relu(layer(x, edge_index, edge_attr))
            else:
                # 前の層の出力を足し込む（Residual）
                x = F.relu(layer(x + residuals[-1], edge_index, edge_attr))
            residuals.append(x)

        # --- プーリング処理 (Mean or Max) ---
        if self.pool_method == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_method == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError("Invalid pool method")
        
        # --- MLPによる予測 ---
        x = F.relu(self.fc1(x))
        action = self.fc2(x)  # 出力: [steer, speed]
        
        return action
""
