// Copyright 2020 Tier IV, Inc.
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

#include "image_projection_based_fusion/roi_cluster_fusion/node.hpp"

#include <image_projection_based_fusion/utils/geometry.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

namespace image_projection_based_fusion
{

RoiClusterFusionNode::RoiClusterFusionNode(const rclcpp::NodeOptions & options)
: FusionNode<DetectedObjectsWithFeature>("roi_cluster_fusion", options)
{
  use_iou_x_ = declare_parameter("use_iou_x", true);
  use_iou_y_ = declare_parameter("use_iou_y", false);
  use_iou_ = declare_parameter("use_iou", false);
  use_cluster_semantic_type_ = declare_parameter("use_cluster_semantic_type", false);
  iou_threshold_ = declare_parameter("iou_threshold", 0.1);
}

void RoiClusterFusionNode::preprocess()
{
  // reset cluster semantic type
  if (!use_cluster_semantic_type_) {
    for (auto & feature_object : output_msg_.feature_objects) {
      feature_object.object.classification.front().label =
        autoware_auto_perception_msgs::msg::ObjectClassification::UNKNOWN;
      feature_object.object.existence_probability = 0.0;
    }
  }
}

void RoiClusterFusionNode::fusionOnSingleImage(
  const int image_id, const DetectedObjectsWithFeature & input_roi_msg,
  const sensor_msgs::msg::CameraInfo & camera_info)
{
  std::vector<sensor_msgs::msg::RegionOfInterest> debug_image_rois;
  std::vector<sensor_msgs::msg::RegionOfInterest> debug_pointcloud_rois;
  std::vector<Eigen::Vector2d> debug_image_points;

  Eigen::Matrix4d projection;
  projection << camera_info.p.at(0), camera_info.p.at(1), camera_info.p.at(2), camera_info.p.at(3),
    camera_info.p.at(4), camera_info.p.at(5), camera_info.p.at(6), camera_info.p.at(7),
    camera_info.p.at(8), camera_info.p.at(9), camera_info.p.at(10), camera_info.p.at(11);

  // get transform from cluster frame id to camera optical frame id
  geometry_msgs::msg::TransformStamped transform_stamped;
  try {
    transform_stamped = tf_buffer_.lookupTransform(
      /*target*/ camera_info.header.frame_id,
      /*src*/ input_msg_.header.frame_id, tf2::TimePointZero);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    return;
  }

  std::map<std::size_t, RegionOfInterest> m_cluster_roi;
  for (std::size_t i = 0; i < input_msg_.feature_objects.size(); ++i) {
    if (input_msg_.feature_objects.at(i).feature.cluster.data.empty()) {
      continue;
    }

    sensor_msgs::msg::PointCloud2 transformed_cluster;
    tf2::doTransform(
      input_msg_.feature_objects.at(i).feature.cluster, transformed_cluster, transform_stamped);

    int min_x(camera_info.width), min_y(camera_info.height), max_x(0), max_y(0);
    std::vector<Eigen::Vector2d> projected_points;
    projected_points.reserve(transformed_cluster.data.size());
    for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(transformed_cluster, "x"),
         iter_y(transformed_cluster, "y"), iter_z(transformed_cluster, "z");
         iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      if (*iter_z <= 0.0) {
        continue;
      }

      Eigen::Vector4d projected_point =
        projection * Eigen::Vector4d(*iter_x, *iter_y, *iter_z, 1.0);
      Eigen::Vector2d normalized_projected_point = Eigen::Vector2d(
        projected_point.x() / projected_point.z(), projected_point.y() / projected_point.z());
      if (
        0 <= static_cast<int>(normalized_projected_point.x()) &&
        static_cast<int>(normalized_projected_point.x()) <=
          static_cast<int>(camera_info.width) - 1 &&
        0 <= static_cast<int>(normalized_projected_point.y()) &&
        static_cast<int>(normalized_projected_point.y()) <=
          static_cast<int>(camera_info.height) - 1) {
        min_x = std::min(static_cast<int>(normalized_projected_point.x()), min_x);
        min_y = std::min(static_cast<int>(normalized_projected_point.y()), min_y);
        max_x = std::max(static_cast<int>(normalized_projected_point.x()), max_x);
        max_y = std::max(static_cast<int>(normalized_projected_point.y()), max_y);
        projected_points.push_back(normalized_projected_point);
        debug_image_points.push_back(normalized_projected_point);
      }
    }
    if (projected_points.empty()) {
      continue;
    }

    sensor_msgs::msg::RegionOfInterest roi;
    // roi.do_rectify = m_camera_info_.at(id).do_rectify;
    roi.x_offset = min_x;
    roi.y_offset = min_y;
    roi.width = max_x - min_x;
    roi.height = max_y - min_y;
    m_cluster_roi.insert(std::make_pair(i, roi));
    debug_pointcloud_rois.push_back(roi);
  }

  for (size_t i = 0; i < input_roi_msg.feature_objects.size(); ++i) {
    int index = 0;
    double max_iou = 0.0;
    for (auto m_cluster_roi_itr = m_cluster_roi.begin(); m_cluster_roi_itr != m_cluster_roi.end();
         ++m_cluster_roi_itr) {
      double iou(0.0), iou_x(0.0), iou_y(0.0);
      if (use_iou_) {
        iou = calcIoU(m_cluster_roi_itr->second, input_roi_msg.feature_objects.at(i).feature.roi);
      }
      if (use_iou_x_) {
        iou_x =
          calcIoUX(m_cluster_roi_itr->second, input_roi_msg.feature_objects.at(i).feature.roi);
      }
      if (use_iou_y_) {
        iou_y =
          calcIoUY(m_cluster_roi_itr->second, input_roi_msg.feature_objects.at(i).feature.roi);
      }
      if (max_iou < iou + iou_x + iou_y) {
        index = m_cluster_roi_itr->first;
        max_iou = iou + iou_x + iou_y;
      }
    }
    if (
      iou_threshold_ < max_iou &&
      output_msg_.feature_objects.at(index).object.existence_probability <=
        input_roi_msg.feature_objects.at(i).object.existence_probability &&
      input_roi_msg.feature_objects.at(i).object.classification.front().label !=
        autoware_auto_perception_msgs::msg::ObjectClassification::UNKNOWN) {
      output_msg_.feature_objects.at(index).object.classification =
        input_roi_msg.feature_objects.at(i).object.classification;
    }
    debug_image_rois.push_back(input_roi_msg.feature_objects.at(i).feature.roi);
  }

  if (debugger_) {
    debugger_->image_rois_ = debug_image_rois;
    debugger_->obstacle_rois_ = debug_pointcloud_rois;
    debugger_->obstacle_points_ = debug_image_points;
    debugger_->publishImage(image_id, input_roi_msg.header.stamp);
  }
}

}  // namespace image_projection_based_fusion

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(image_projection_based_fusion::RoiClusterFusionNode)
