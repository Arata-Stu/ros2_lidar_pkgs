cmake_minimum_required(VERSION 3.8)
project(lidar_graph_msgs)

# 必要な依存パッケージ
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# カスタムメッセージの追加
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GraphData.msg"
)

# 必要な依存関係
ament_export_dependencies(rosidl_default_runtime)
ament_package()
