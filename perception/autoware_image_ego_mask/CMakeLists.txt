cmake_minimum_required(VERSION 3.14)
project(autoware_image_ego_mask)

find_package(ament_cmake_auto REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yaml-cpp REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/ego_mask_node.cpp
)

ament_target_dependencies(${PROJECT_NAME} cv_bridge)
target_link_libraries(${PROJECT_NAME} yaml-cpp)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "autoware::image_preprocessor::EgoMaskNode"
  EXECUTABLE ego_mask_node
)

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)

