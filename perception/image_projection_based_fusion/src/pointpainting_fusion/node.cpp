// Copyright 2022 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "image_projection_based_fusion/pointpainting_fusion/node.hpp"

#include <image_projection_based_fusion/utils/geometry.hpp>
#include <image_projection_based_fusion/utils/utils.hpp>
#include <lidar_centerpoint/centerpoint_config.hpp>
#include <lidar_centerpoint/preprocess/pointcloud_densification.hpp>
#include <lidar_centerpoint/ros_utils.hpp>
#include <lidar_centerpoint/utils.hpp>
#include <tier4_autoware_utils/geometry/geometry.hpp>
#include <tier4_autoware_utils/math/constants.hpp>

namespace image_projection_based_fusion
{

PointpaintingFusionNode::PointpaintingFusionNode(const rclcpp::NodeOptions & options)
: FusionNode<sensor_msgs::msg::PointCloud2, DetectedObjects>("pointpainting_fusion", options)
{
  //   const float score_threshold =
  //     static_cast<float>(this->declare_parameter<double>("score_threshold", 0.4));
  //   const float circle_nms_dist_threshold =
  //     static_cast<float>(this->declare_parameter<double>("circle_nms_dist_threshold", 1.5));
  //   // densification param
  //   const std::string densification_world_frame_id =
  //     this->declare_parameter("densification_world_frame_id", "map");
  //   const int densification_num_past_frames =
  //     this->declare_parameter("densification_num_past_frames", 0);
  //   // network param
  //   const std::string trt_precision = this->declare_parameter("trt_precision", "fp16");
  //   const std::string encoder_onnx_path = this->declare_parameter("encoder_onnx_path", "");
  //   const std::string encoder_engine_path = this->declare_parameter("encoder_engine_path", "");
  //   const std::string head_onnx_path = this->declare_parameter("head_onnx_path", "");
  //   const std::string head_engine_path = this->declare_parameter("head_engine_path", "");
  //   class_names_ = this->declare_parameter<std::vector<std::string>>("class_names");
  //   rename_car_to_truck_and_bus_ = this->declare_parameter("rename_car_to_truck_and_bus", false);
  //   has_twist_ = this->declare_parameter("has_twist", false);
  //   const std::size_t point_feature_size =
  //     static_cast<std::size_t>(this->declare_parameter<std::int64_t>("point_feature_size"));
  //   const std::size_t max_voxel_size =
  //     static_cast<std::size_t>(this->declare_parameter<std::int64_t>("max_voxel_size"));
  pointcloud_range = this->declare_parameter<std::vector<double>>("point_cloud_range");
  //   const auto voxel_size = this->declare_parameter<std::vector<double>>("voxel_size");
  //   const std::size_t downsample_factor =
  //     static_cast<std::size_t>(this->declare_parameter<std::int64_t>("downsample_factor"));
  //   const std::size_t encoder_in_feature_size =
  //     static_cast<std::size_t>(this->declare_parameter<std::int64_t>("encoder_in_feature_size"));

  std::function<void(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)> sub_callback =
    std::bind(&PointpaintingFusionNode::subCallback, this, std::placeholders::_1);
  sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "~/input/pointcloud", rclcpp::SensorDataQoS().keep_last(3), sub_callback);
}

void PointpaintingFusionNode::preprocess(sensor_msgs::msg::PointCloud2 & painted_pointcloud_msg)
{
  sensor_msgs::msg::PointCloud2 tmp;
  tmp = painted_pointcloud_msg;

  // set fields
  sensor_msgs::PointCloud2Modifier pcd_modifier(painted_pointcloud_msg);
  pcd_modifier.clear();
  painted_pointcloud_msg.width = tmp.width;
  constexpr int num_fields = 7;
  pcd_modifier.setPointCloud2Fields(
    num_fields, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
    sensor_msgs::msg::PointField::FLOAT32, "z", 1, sensor_msgs::msg::PointField::FLOAT32,
    "intensity", 1, sensor_msgs::msg::PointField::FLOAT32, "CAR", 1,
    sensor_msgs::msg::PointField::FLOAT32, "PEDESTRIAN", 1, sensor_msgs::msg::PointField::FLOAT32,
    "BICYCLE", 1, sensor_msgs::msg::PointField::FLOAT32);
  painted_pointcloud_msg.point_step = num_fields * sizeof(float);

  // filter points out of range
  const auto painted_point_step = painted_pointcloud_msg.point_step;
  size_t j = 0;
  sensor_msgs::PointCloud2Iterator<float> iter_painted_x(painted_pointcloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_painted_y(painted_pointcloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_painted_z(painted_pointcloud_msg, "z");
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(tmp, "x"), iter_y(tmp, "y"),
       iter_z(tmp, "z");
       iter_x != iter_x.end();
       ++iter_x, ++iter_y, ++iter_z, ++iter_painted_x, ++iter_painted_y, ++iter_painted_z) {
    if (
      *iter_x <= pointcloud_range.at(0) || *iter_x >= pointcloud_range.at(3) ||
      *iter_y <= pointcloud_range.at(1) || *iter_y >= pointcloud_range.at(4)) {
      continue;
    } else {
      *iter_painted_x = *iter_x;
      *iter_painted_y = *iter_y;
      *iter_painted_z = *iter_z;
      j += painted_point_step;
    }
  }
  painted_pointcloud_msg.data.resize(j);
  painted_pointcloud_msg.width = static_cast<uint32_t>(
    painted_pointcloud_msg.data.size() / painted_pointcloud_msg.height /
    painted_pointcloud_msg.point_step);
  painted_pointcloud_msg.row_step =
    static_cast<uint32_t>(painted_pointcloud_msg.data.size() / painted_pointcloud_msg.height);
}

void PointpaintingFusionNode::fuseOnSingleImage(
  __attribute__((unused)) const sensor_msgs::msg::PointCloud2 & input_pointcloud_msg,
  const std::size_t image_id __attribute__((unused)),
  const DetectedObjectsWithFeature & input_roi_msg __attribute__((unused)),
  const sensor_msgs::msg::CameraInfo & camera_info __attribute__((unused)),
  sensor_msgs::msg::PointCloud2 & painted_pointcloud_msg __attribute__((unused)))
{
  std::vector<sensor_msgs::msg::RegionOfInterest> debug_image_rois;
  std::vector<Eigen::Vector2d> debug_image_points;

  // get transform from cluster frame id to camera optical frame id
  geometry_msgs::msg::TransformStamped transform_stamped;
  {
    const auto transform_stamped_optional = getTransformStamped(
      tf_buffer_, /*target*/ camera_info.header.frame_id,
      /*source*/ painted_pointcloud_msg.header.frame_id, camera_info.header.stamp);
    if (!transform_stamped_optional) {
      return;
    }
    transform_stamped = transform_stamped_optional.value();
  }

  // projection matrix
  Eigen::Matrix4d camera_projection;
  camera_projection << camera_info.p.at(0), camera_info.p.at(1), camera_info.p.at(2),
    camera_info.p.at(3), camera_info.p.at(4), camera_info.p.at(5), camera_info.p.at(6),
    camera_info.p.at(7), camera_info.p.at(8), camera_info.p.at(9), camera_info.p.at(10),
    camera_info.p.at(11);

  // transform
  sensor_msgs::msg::PointCloud2 transformed_pointcloud;
  tf2::doTransform(painted_pointcloud_msg, transformed_pointcloud, transform_stamped);

  // iterate points
  sensor_msgs::PointCloud2Iterator<float> iter_painted_intensity(
    painted_pointcloud_msg, "intensity");
  sensor_msgs::PointCloud2Iterator<float> iter_car(painted_pointcloud_msg, "CAR");
  sensor_msgs::PointCloud2Iterator<float> iter_ped(painted_pointcloud_msg, "PEDESTRIAN");
  sensor_msgs::PointCloud2Iterator<float> iter_bic(painted_pointcloud_msg, "BICYCLE");
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(transformed_pointcloud, "x"),
       iter_y(transformed_pointcloud, "y"), iter_z(transformed_pointcloud, "z");
       iter_x != iter_x.end();
       ++iter_x, ++iter_y, ++iter_z, ++iter_painted_intensity, ++iter_car, ++iter_ped, ++iter_bic) {
    if (*iter_z <= 0.0) {
      continue;
    }
    // project
    Eigen::Vector4d projected_point =
      camera_projection * Eigen::Vector4d(*iter_x, *iter_y, *iter_z, 1.0);
    Eigen::Vector2d normalized_projected_point = Eigen::Vector2d(
      projected_point.x() / projected_point.z(), projected_point.y() / projected_point.z());

    // iterate 2d bbox
    for (const auto & feature_object : input_roi_msg.feature_objects) {
      sensor_msgs::msg::RegionOfInterest roi = feature_object.feature.roi;
      // paint current point if it is inside bbox
      if (
        normalized_projected_point.x() >= roi.x_offset &&
        normalized_projected_point.x() <= roi.x_offset + roi.width &&
        normalized_projected_point.y() >= roi.y_offset &&
        normalized_projected_point.y() <= roi.y_offset + roi.height &&
        feature_object.object.classification.front().label !=
          autoware_auto_perception_msgs::msg::ObjectClassification::UNKNOWN) {
        switch (feature_object.object.classification.front().label) {
          case autoware_auto_perception_msgs::msg::ObjectClassification::CAR:
            *iter_car = 1.0;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::TRUCK:
            *iter_car = 1.0;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::TRAILER:
            *iter_car = 1.0;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::BUS:
            *iter_car = 1.0;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::PEDESTRIAN:
            *iter_ped = 1.0;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::BICYCLE:
            *iter_bic = 1.0;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::MOTORCYCLE:
            *iter_bic = 1.0;
            break;
        }
        // debug_image_points.push_back(normalized_projected_point);
      }
    }
  }
  // for (const auto & feature_object : input_roi_msg.feature_objects) {
  //   debug_image_rois.push_back(feature_object.feature.roi);
  // }

  // if (debugger_) {
  //   debugger_->image_rois_ = debug_image_rois;
  //   debugger_->obstacle_points_ = debug_image_points;
  //   debugger_->publishImage(image_id, input_roi_msg.header.stamp);
  // }
}

// void PointpaintingFusionNode::postprocess(sensor_msgs::msg::PointCloud2 & painted_pointcloud_msg)
// {}

}  // namespace image_projection_based_fusion

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(image_projection_based_fusion::PointpaintingFusionNode)