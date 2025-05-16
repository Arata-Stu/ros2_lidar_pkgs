cmake_minimum_required(VERSION 3.5)
project(scan2graph)

# 使用するC++のバージョン
set(CMAKE_CXX_STANDARD 17)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(lidar_graph_msgs REQUIRED)



# ソースファイルの指定
add_executable(lidar_graph_node src/lidar_graph.cpp)
ament_target_dependencies(lidar_graph_node
    rclcpp
    sensor_msgs
    std_msgs
    visualization_msgs 
    lidar_graph_msgs 
)

# インストールターゲットの指定
install(TARGETS lidar_graph_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
