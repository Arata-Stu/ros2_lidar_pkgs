import rclpy
from rclpy.node import Node
from lidar_graph_msgs.msg import GraphData
from ament_index_python.packages import get_package_share_directory
import torch
from torch_geometric.data import Data
import yaml
import os

from gnn_node.models import build_model


class GNNnode(Node):
    def __init__(self):
        super().__init__('gnn_node')

        # パラメータの取得
        self.model_name = self.get_parameter('model_name').value
        self.input_dim = self.get_parameter('input_dim').value
        self.output_dim = self.get_parameter('output_dim').value
        self.hidden_dim = self.get_parameter('hidden_dim').value
        self.debug_mode = self.get_parameter('debug_mode').value

        # モデルの初期化
        self.model = build_model(self.model_name, self.input_dim, self.hidden_dim, self.output_dim)

        self.subscription = self.create_subscription(
            GraphData,
            '/lidar_graph',
            self.graph_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info(f"gnn_node has been started with model: {self.model_name}")

    def graph_callback(self, msg):
        self.get_logger().info(f"GraphData received: Nodes={len(msg.node_x)}, Edges={len(msg.edge_from)}")

        # ノードの座標をTorchのテンソルに変換
        x = torch.tensor(list(zip(msg.node_x, msg.node_y)), dtype=torch.float)

        # エッジインデックスをTorchのテンソルに変換
        edge_index = torch.tensor([msg.edge_from, msg.edge_to], dtype=torch.long)

        # エッジの重みも存在する場合はテンソルに変換
        if len(msg.edge_weight) > 0:
            edge_attr = torch.tensor(msg.edge_weight, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            data = Data(x=x, edge_index=edge_index)

        # デバッグモードなら詳細情報を出力
        if self.debug_mode:
            self.get_logger().info(f"Data object created: {data}")
            self.get_logger().info(f"Number of Nodes: {data.num_nodes}")
            self.get_logger().info(f"Number of Edges: {data.num_edges}")
            self.get_logger().info(f"Edge Index:\n{data.edge_index}")

        # モデルの推論
        output = self.model(data)
        self.get_logger().info(f"Model output: {output}")


def main(args=None):
    rclpy.init(args=args)
    node = GNNnode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
