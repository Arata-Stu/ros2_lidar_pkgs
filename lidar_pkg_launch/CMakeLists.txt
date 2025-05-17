cmake_minimum_required(VERSION 3.8)
project(lidar_pkg_launch)

# 必要な依存パッケージ
find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})  


ament_package()
