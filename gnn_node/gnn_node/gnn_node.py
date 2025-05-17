import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from lidar_graph_msgs.msg import GraphData
from ackermann_msgs.msg import AckermannDrive
import torch
from torch_geometric.data import Data

from gnn_node.models import build_model


class GNNnode(Node):
    def __init__(self):
        super().__init__('gnn_node')

        # パラメータ宣言
        self.declare_parameter('model_name', 'default_model')
        self.declare_parameter('input_dim', 2)
        self.declare_parameter('hidden_dim', 64)
        self.declare_parameter('output_dim', 2)
        self.declare_parameter('debug_mode', False)

        # パラメータ取得
        self._load_parameters()

        # モデルの初期化
        self._build_model()

        # 動的パラメータ変更コールバック登録
        self.add_on_set_parameters_callback(self._on_param_change)

        # トピック設定
        self.subscription = self.create_subscription(
            GraphData, '/lidar_graph', self.graph_callback, 10)
        self.publisher = self.create_publisher(AckermannDrive, '/cmd_drive', 10)
        self.get_logger().info(f"gnn_node 起動: model={self.model_name}")

    def _load_parameters(self):
        p = self.get_parameters([
            'model_name', 'input_dim', 'hidden_dim', 'output_dim', 'debug_mode'
        ])
        # パラメータ名と属性を対応づけ
        for param in p:
            setattr(self, param.name, param.value)

    def _build_model(self):
        # もし GPU のキャッシュが気になるなら空にする
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # 古いモデルを破棄
        if hasattr(self, 'model'):
            del self.model
        # 新しいモデルを構築
        self.model = build_model(
            self.model_name,
            self.input_dim,
            self.hidden_dim,
            self.output_dim
        )
        self.get_logger().info(f"[MODEL] 再構築: {self.model_name}, hidden={self.hidden_dim}")

    def _on_param_change(self, params):
        # パラメータ変更時に呼ばれるコールバック
        rebuild_needed = False
        for param in params:
            if param.name in ('model_name', 'input_dim', 'hidden_dim', 'output_dim'):
                # 型チェック（省略可）
                setattr(self, param.name, param.value)
                rebuild_needed = True
            elif param.name == 'debug_mode':
                self.debug_mode = param.value

        if rebuild_needed:
            self._build_model()

        # 結果を返す
        result = SetParametersResult()
        result.successful = True
        return result

    def graph_callback(self, msg):
        # …（既存の callback 実装）…
        x = torch.tensor(list(zip(msg.node_x, msg.node_y)), dtype=torch.float)
        edge_index = torch.tensor([msg.edge_from, msg.edge_to], dtype=torch.long)
        if msg.edge_weight:
            data = Data(x=x, edge_index=edge_index,
                        edge_attr=torch.tensor(msg.edge_weight, dtype=torch.float))
        else:
            data = Data(x=x, edge_index=edge_index)

        if self.debug_mode:
            self.get_logger().info(f"[DEBUG] data={data}")

        output = self.model(data)
        steer = output[0, 0].clamp(-1.0, 1.0).item()
        speed = output[0, 1].clamp(0.0, 1.0).item()

        drive_msg = AckermannDrive()
        drive_msg.steering_angle = steer
        drive_msg.speed = speed
        self.publisher.publish(drive_msg)
        self.get_logger().info(f"Published: steer={steer:.3f}, speed={speed:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = GNNnode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
