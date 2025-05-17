from .gnn import N4LidarGCN

def build_model(model_name, input_dim, hidden_dim, output_dim, pool_method="mean"):
    """
    モデルをビルドして返します。

    Args:
        model_name (str): 使用するモデルの名前
        input_dim (int): 入力次元
        hidden_dim (int): 隠れ層の次元
        output_dim (int): 出力次元

    Returns:
        torch.nn.Module: 指定したモデルのインスタンス
    """
    if model_name == "n4_lidar_gcn":
        return N4LidarGCN(input_dim, hidden_dim, output_dim, pool_method)
    else:
        raise ValueError(f"指定されたモデル名 '{model_name}' は存在しません。")