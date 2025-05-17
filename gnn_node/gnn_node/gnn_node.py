import rclpy
from rclpy.node import Node
from lidar_graph_msgs.msg import GraphData
from ackermann_msgs.msg import AckermannDrive
from ament_index_python.packages import get_package_share_directory
import torch
from torch_geometric.data import Data

from gnn_node.models import build_model


class GNNnode(Node):
    def __init__(self):
        super().__init__('gnn_node')

        # パラメータの宣言
        self.declare_parameter('model_name', 'default_model')
        self.declare_parameter('input_dim', 2)
        self.declare_parameter('output_dim', 2)  # steer, speed
        self.declare_parameter('hidden_dim', 64)
        self.declare_parameter('debug_mode', False)

        # パラメータの取得
        self.model_name = self.get_parameter('model_name').value
        self.input_dim = self.get_parameter('input_dim').value
        self.output_dim = self.get_parameter('output_dim').value
        self.hidden_dim = self.get_parameter('hidden_dim').value
        self.debug_mode = self.get_parameter('debug_mode').value

        # モデルの初期化
        self.model = build_model(self.model_name, self.input_dim, self.hidden_dim, self.output_dim)

        # トピックの購読設定
        self.subscription = self.create_subscription(
            GraphData,
            '/lidar_graph',
            self.graph_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info(f"gnn_node has been started with model: {self.model_name}")

        # パブリッシャーの初期化
        self.publisher = self.create_publisher(AckermannDrive, '/cmd_drive', 10)

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

        # 出力の確認
        self.get_logger().info(f"Model output: {output}")

        # AckermannDrive メッセージの作成
        drive_msg = AckermannDrive()

        # steer と speed の設定
        drive_msg.steering_angle = float(max(min(output[0].item(), 1.0), -1.0))
        drive_msg.speed = float(max(min(output[1].item(), 1.0), -1.0))

        # パブリッシュ
        self.publisher.publish(drive_msg)
        self.get_logger().info(f"Published AckermannDrive: Steering={drive_msg.steering_angle}, Speed={drive_msg.speed}")


def main(args=None):
    rclpy.init(args=args)
    node = GNNnode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
