import rclpy
from rclpy.node import Node
from lidar_graph_msgs.msg import GraphData
import torch
from torch_geometric.data import Data


class GraphSubscriberNode(Node):
    def __init__(self):
        super().__init__('graph_subscriber_node')
        self.subscription = self.create_subscription(
            GraphData,
            '/lidar_graph',
            self.graph_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info("GraphSubscriberNode has been started.")

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

        # デバッグ表示
        self.get_logger().info(f"Data object created: {data}")
        self.get_logger().info(f"Number of Nodes: {data.num_nodes}")
        self.get_logger().info(f"Number of Edges: {data.num_edges}")
        self.get_logger().info(f"Edge Index:\n{data.edge_index}")

        # TODO: ここでGNNへの処理を追加する

def main(args=None):
    rclpy.init(args=args)
    node = GraphSubscriberNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
