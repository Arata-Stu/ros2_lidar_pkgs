#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <lidar_graph_msgs/msg/graph_data.hpp>
#include <vector>
#include <cmath>

#define DEG2RAD (M_PI / 180.0)
#define MAX_EDGES_PER_NODE 5

class GraphNode {
public:
    double x, y;
    GraphNode(double x = 0, double y = 0) : x(x), y(y) {}
};

class Edge {
public:
    int from, to;
    double weight;
    Edge(int from = 0, int to = 0, double weight = 0.0)
        : from(from), to(to), weight(weight) {}
};

class LidarGraphNode : public rclcpp::Node {
public:
    LidarGraphNode() : rclcpp::Node("lidar_graph_node") {
        // パラメータの宣言
        this->declare_parameter<double>("edge_threshold", 1.0);
        this->declare_parameter<double>("fov", 270.0);
        this->declare_parameter<bool>("debug_mode", false);

        // パラメータの取得
        this->get_parameter("edge_threshold", edge_threshold_);
        this->get_parameter("fov", fov_);
        this->get_parameter("debug_mode", debug_mode_);

        subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&LidarGraphNode::scan_callback, this, std::placeholders::_1)
        );

        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/lidar_edges", 10
        );

        graph_publisher_ = this->create_publisher<lidar_graph_msgs::msg::GraphData>(
            "/lidar_graph", 10
        );

        RCLCPP_INFO(this->get_logger(), "LidarGraphNode has been started.");
        if (debug_mode_) {
            RCLCPP_INFO(this->get_logger(), "Debug Mode: Enabled");
        }
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // 最新のパラメータ値を取得
        this->get_parameter("edge_threshold", edge_threshold_);
        this->get_parameter("fov", fov_);
        this->get_parameter("debug_mode", debug_mode_);

        int num_nodes = msg->ranges.size();
        double angle_increment = fov_ / num_nodes;

        nodes.clear();
        edges.clear();
        edge_counts = 0;

        if (debug_mode_) {
            RCLCPP_INFO(this->get_logger(), "Number of nodes: %d", num_nodes);
            RCLCPP_INFO(this->get_logger(), "Angle increment: %f", angle_increment);
        }

        // ノードの位置を計算
        for (int i = 0; i < num_nodes; ++i) {
            double dist = msg->ranges[i];
            double angle = (-fov_ / 2.0 + i * angle_increment) * DEG2RAD;
            nodes.emplace_back(dist * cos(angle), dist * sin(angle));

            if (debug_mode_) {
                RCLCPP_INFO(this->get_logger(), "Node[%d]: (x: %f, y: %f)", i, nodes.back().x, nodes.back().y);
            }
        }

        // エッジの構築
        for (int i = 0; i < num_nodes - 1; ++i) {
            double dx = nodes[i + 1].x - nodes[i].x;
            double dy = nodes[i + 1].y - nodes[i].y;
            double d = std::sqrt(dx * dx + dy * dy);
            if (d < edge_threshold_) {
                edges.emplace_back(i, i + 1, d);

                if (debug_mode_) {
                    RCLCPP_INFO(this->get_logger(), "Edge[%d]: (%d -> %d), Distance: %f", edge_counts, i, i + 1, d);
                }
            }
        }

        // データをパブリッシュ
        publish_edges();
        publish_graph_data();
    }

    void publish_edges() {
        visualization_msgs::msg::MarkerArray marker_array;

        for (const auto &edge : edges) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "laser";
            marker.header.stamp = this->now();
            marker.ns = "lidar_edges";
            marker.id = edge_counts++;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.02;
            marker.color.a = 1.0;
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;

            geometry_msgs::msg::Point p1, p2;
            p1.x = nodes[edge.from].x;
            p1.y = nodes[edge.from].y;
            p2.x = nodes[edge.to].x;
            p2.y = nodes[edge.to].y;

            marker.points.push_back(p1);
            marker.points.push_back(p2);

            marker_array.markers.push_back(marker);
        }

        marker_publisher_->publish(marker_array);
    }

    void publish_graph_data() {
        lidar_graph_msgs::msg::GraphData msg;

        // ノード座標
        for (const auto &node : nodes) {
            msg.node_x.push_back(node.x);
            msg.node_y.push_back(node.y);
        }

        // エッジ情報
        for (const auto &edge : edges) {
            msg.edge_from.push_back(edge.from);
            msg.edge_to.push_back(edge.to);
            msg.edge_weight.push_back(edge.weight);
        }

        graph_publisher_->publish(msg);

        if (debug_mode_) {
            RCLCPP_INFO(this->get_logger(), "GraphData Published: Nodes=%ld, Edges=%ld", msg.node_x.size(), msg.edge_from.size());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::Publisher<lidar_graph_msgs::msg::GraphData>::SharedPtr graph_publisher_;
    std::vector<GraphNode> nodes;
    std::vector<Edge> edges;
    int edge_counts;

    // 動的パラメータ
    double edge_threshold_;
    double fov_;
    bool debug_mode_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarGraphNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
