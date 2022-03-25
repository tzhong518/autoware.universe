// Copyright 2022 Tier IV, Inc.
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

#include "image_projection_based_fusion/roi_detected_object_fusion/node.hpp"

#include <image_projection_based_fusion/utils/geometry.hpp>
#include <image_projection_based_fusion/utils/utils.hpp>

namespace image_projection_based_fusion
{

RoiDetectedObjectFusionNode::RoiDetectedObjectFusionNode(const rclcpp::NodeOptions & options)
: FusionNode<DetectedObjects>("roi_detected_object_fusion", options)
{
  use_iou_x_ = declare_parameter("use_iou_x", false);
  use_iou_y_ = declare_parameter("use_iou_y", false);
  use_iou_ = declare_parameter("use_iou", false);
  iou_threshold_ = declare_parameter("iou_threshold", 0.1);
}

void RoiDetectedObjectFusionNode::fusionOnSingleImage(
  const int image_id, const DetectedObjectsWithFeature & input_roi_msg,
  const sensor_msgs::msg::CameraInfo & camera_info)
{
  // TODO(yukke42): define getTransformStamped
  // TODO(yukke42): set the transform stamp
  geometry_msgs::msg::TransformStamped transform_stamped;
  try {
    transform_stamped = tf_buffer_.lookupTransform(
      /*target*/ camera_info.header.frame_id,
      /*src*/ input_msg_.header.frame_id, tf2::TimePointZero);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    return;
  }

  Eigen::Affine3d object2camera_affine = transformToEigen(transform_stamped.transform);

  // TODO(yukke42): define getProjectionMatrix
  Eigen::Matrix4d camera_projection;
  camera_projection << camera_info.p.at(0), camera_info.p.at(1), camera_info.p.at(2),
    camera_info.p.at(3), camera_info.p.at(4), camera_info.p.at(5), camera_info.p.at(6),
    camera_info.p.at(7), camera_info.p.at(8), camera_info.p.at(9), camera_info.p.at(10),
    camera_info.p.at(11);

  std::map<std::size_t, sensor_msgs::msg::RegionOfInterest> object_roi_map;
  generateDetectedObjectRois(
    static_cast<double>(camera_info.width), static_cast<double>(camera_info.height),
    object2camera_affine, camera_projection, object_roi_map);
  updateDetectedObjectClassification(
    input_roi_msg.feature_objects, object_roi_map, output_msg_.objects);

  if (debugger_) {
    debugger_->image_rois_.reserve(input_roi_msg.feature_objects.size());
    for (std::size_t roi_i = 0; roi_i < input_roi_msg.feature_objects.size(); roi_i++) {
      debugger_->image_rois_.push_back(input_roi_msg.feature_objects.at(roi_i).feature.roi);
    }
    debugger_->publishImage(image_id, input_roi_msg.header.stamp);
  }
}

void RoiDetectedObjectFusionNode::generateDetectedObjectRois(
  const double image_width, const double image_height, const Eigen::Affine3d & object2camera_affine,
  const Eigen::Matrix4d & camera_projection,
  std::map<std::size_t, sensor_msgs::msg::RegionOfInterest> & object_roi_map)
{
  for (std::size_t obj_i = 0; obj_i < input_msg_.objects.size(); obj_i++) {
    std::vector<Eigen::Vector3d> vertices_camera_coord;
    {
      const auto & object = input_msg_.objects.at(obj_i);
      std::vector<Eigen::Vector3d> vertices;
      objectToVertices(object.kinematics.pose_with_covariance.pose, object.shape, vertices);
      transformPoints(vertices, object2camera_affine, vertices_camera_coord);
    }

    double min_x(image_width), min_y(image_height), max_x(0.0), max_y(0.0);
    std::size_t point_on_image_cnt = 0;
    for (const auto & point : vertices_camera_coord) {
      if (point.z() <= 0.0) {
        continue;
      }

      Eigen::Vector2d proj_point;
      {
        Eigen::Vector4d proj_point_hom =
          camera_projection * Eigen::Vector4d(point.x(), point.y(), point.z(), 1.0);
        proj_point = Eigen::Vector2d(
          proj_point_hom.x() / (proj_point_hom.z()), proj_point_hom.y() / (proj_point_hom.z()));
      }
      // TODO(yukke42): consider the outside point of the object on image.
      if (
        proj_point.x() >= 0 && proj_point.x() <= image_width - 1 && proj_point.y() >= 0 &&
        proj_point.y() <= image_height - 1) {
        point_on_image_cnt++;

        min_x = std::min(proj_point.x(), min_x);
        min_y = std::min(proj_point.y(), min_y);
        max_x = std::max(proj_point.x(), max_x);
        max_y = std::max(proj_point.y(), max_y);

        if (debugger_) {
          debugger_->obstacle_points_.push_back(proj_point);
        }
      }
    }
    if (point_on_image_cnt == 0) {
      continue;
    }

    // build roi
    sensor_msgs::msg::RegionOfInterest roi;
    roi.x_offset = static_cast<std::uint32_t>(min_x);
    roi.y_offset = static_cast<std::uint32_t>(min_y);
    roi.width = static_cast<std::uint32_t>(max_x - min_x);
    roi.height = static_cast<std::uint32_t>(max_y - min_y);
    object_roi_map.insert(std::make_pair(obj_i, roi));

    if (debugger_) {
      debugger_->obstacle_rois_.push_back(roi);
    }
  }
}

void RoiDetectedObjectFusionNode::updateDetectedObjectClassification(
  const std::vector<DetectedObjectWithFeature> & image_rois,
  const std::map<std::size_t, sensor_msgs::msg::RegionOfInterest> & object_roi_map,
  std::vector<DetectedObject> & output_objects)
{
  for (std::size_t roi_i = 0; roi_i < image_rois.size(); roi_i++) {
    const auto & roi = image_rois.at(roi_i).feature.roi;
    std::size_t object_i = 0;
    double max_iou = 0.0;
    for (auto object_itr = object_roi_map.begin(); object_itr != object_roi_map.end();
         object_itr++) {
      double iou(0.0), iou_x(0.0), iou_y(0.0);
      if (use_iou_) {
        iou = calcIoU(object_itr->second, roi);
      }
      if (use_iou_x_) {
        iou_x = calcIoUX(object_itr->second, roi);
      }
      if (use_iou_y_) {
        iou_y = calcIoUY(object_itr->second, roi);
      }

      if (iou + iou_x + iou_y > max_iou) {
        object_i = object_itr->first;
        max_iou = iou + iou_x + iou_y;
      }
    }

    if (
      max_iou > iou_threshold_ && output_objects.at(object_i).existence_probability <=
                                    image_rois.at(roi_i).object.existence_probability) {
      output_objects.at(object_i).classification = image_rois.at(roi_i).object.classification;
    }
  }
}

}  // namespace image_projection_based_fusion

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(image_projection_based_fusion::RoiDetectedObjectFusionNode)
