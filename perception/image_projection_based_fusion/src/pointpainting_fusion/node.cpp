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

#include "autoware_point_types/types.hpp"

#include <image_projection_based_fusion/utils/geometry.hpp>
#include <image_projection_based_fusion/utils/utils.hpp>
#include <lidar_centerpoint/centerpoint_config.hpp>
#include <lidar_centerpoint/preprocess/pointcloud_densification.hpp>
#include <lidar_centerpoint/ros_utils.hpp>
#include <lidar_centerpoint/utils.hpp>
#include <tier4_autoware_utils/geometry/geometry.hpp>
#include <tier4_autoware_utils/math/constants.hpp>

#include <omp.h>

#include <chrono>

static double t_sum = 0.0;
static int t_counter = 0;
namespace image_projection_based_fusion
{

inline bool isInsideBbox(
  float proj_x, float proj_y, sensor_msgs::msg::RegionOfInterest roi, float zc)
{
  return proj_x >= roi.x_offset * zc && proj_x <= (roi.x_offset + roi.width) * zc &&
         proj_y >= roi.y_offset * zc && proj_y <= (roi.y_offset + roi.height) * zc;
}

inline bool isVehicle(int label2d)
{
  return label2d == autoware_auto_perception_msgs::msg::ObjectClassification::CAR;
}

inline bool isTruck(int label2d)
{
  return label2d == autoware_auto_perception_msgs::msg::ObjectClassification::TRUCK ||
         label2d == autoware_auto_perception_msgs::msg::ObjectClassification::TRAILER;
}

inline bool isBus(int label2d)
{
  return label2d == autoware_auto_perception_msgs::msg::ObjectClassification::BUS;
}

inline bool isPedestrian(int label2d)
{
  return label2d == autoware_auto_perception_msgs::msg::ObjectClassification::PEDESTRIAN;
}

inline bool isBicycle(int label2d)
{
  return label2d == autoware_auto_perception_msgs::msg::ObjectClassification::BICYCLE ||
         label2d == autoware_auto_perception_msgs::msg::ObjectClassification::MOTORCYCLE;
}

inline bool isUnknown(int label2d)
{
  return label2d == autoware_auto_perception_msgs::msg::ObjectClassification::UNKNOWN;
}

PointPaintingFusionNode::PointPaintingFusionNode(const rclcpp::NodeOptions & options)
: FusionNode<sensor_msgs::msg::PointCloud2, DetectedObjects>("pointpainting_fusion", options)
{
  const float score_threshold =
    static_cast<float>(this->declare_parameter<double>("score_threshold", 0.4));
  const float circle_nms_dist_threshold =
    static_cast<float>(this->declare_parameter<double>("circle_nms_dist_threshold", 1.5));
  const auto yaw_norm_thresholds =
    this->declare_parameter<std::vector<double>>("yaw_norm_thresholds");
  // densification param
  const std::string densification_world_frame_id =
    this->declare_parameter("densification_world_frame_id", "map");
  const int densification_num_past_frames =
    this->declare_parameter("densification_num_past_frames", 0);
  // network param
  const std::string trt_precision = this->declare_parameter("trt_precision", "fp16");
  const std::string encoder_onnx_path = this->declare_parameter("encoder_onnx_path", "");
  const std::string encoder_engine_path = this->declare_parameter("encoder_engine_path", "");
  const std::string head_onnx_path = this->declare_parameter("head_onnx_path", "");
  const std::string head_engine_path = this->declare_parameter("head_engine_path", "");
  class_names_ = this->declare_parameter<std::vector<std::string>>("class_names");
  has_twist_ = this->declare_parameter("has_twist", false);
  const std::size_t point_feature_size =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("point_feature_size"));
  const std::size_t max_voxel_size =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("max_voxel_size"));
  pointcloud_range = this->declare_parameter<std::vector<double>>("point_cloud_range");
  const auto voxel_size = this->declare_parameter<std::vector<double>>("voxel_size");
  const std::size_t downsample_factor =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("downsample_factor"));
  const std::size_t encoder_in_feature_size =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("encoder_in_feature_size"));
  const auto allow_remapping_by_area_matrix =
    this->declare_parameter<std::vector<int64_t>>("allow_remapping_by_area_matrix");
  const auto min_area_matrix = this->declare_parameter<std::vector<double>>("min_area_matrix");
  const auto max_area_matrix = this->declare_parameter<std::vector<double>>("max_area_matrix");

  std::function<void(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)> sub_callback =
    std::bind(&PointPaintingFusionNode::subCallback, this, std::placeholders::_1);
  sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "~/input/pointcloud", rclcpp::SensorDataQoS().keep_last(3), sub_callback);

  tan_h_.resize(rois_number_);
  for (std::size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    auto fx = camera_info_map_[roi_i].k.at(0);
    auto x0 = camera_info_map_[roi_i].k.at(2);
    tan_h_[roi_i] = x0 / fx;
  }

  detection_class_remapper_.setParameters(
    allow_remapping_by_area_matrix, min_area_matrix, max_area_matrix);

  centerpoint::NetworkParam encoder_param(encoder_onnx_path, encoder_engine_path, trt_precision);
  centerpoint::NetworkParam head_param(head_onnx_path, head_engine_path, trt_precision);
  centerpoint::DensificationParam densification_param(
    densification_world_frame_id, densification_num_past_frames);
  centerpoint::CenterPointConfig config(
    class_names_.size(), point_feature_size, max_voxel_size, pointcloud_range, voxel_size,
    downsample_factor, encoder_in_feature_size, score_threshold, circle_nms_dist_threshold,
    yaw_norm_thresholds);

  // create detector
  detector_ptr_ = std::make_unique<image_projection_based_fusion::PointPaintingTRT>(
    encoder_param, head_param, densification_param, config);

  obj_pub_ptr_ = this->create_publisher<DetectedObjects>("~/output/objects", rclcpp::QoS{1});
}

void PointPaintingFusionNode::preprocess(sensor_msgs::msg::PointCloud2 & painted_pointcloud_msg)
{
  sensor_msgs::msg::PointCloud2 tmp;
  tmp = painted_pointcloud_msg;

  // set fields
  sensor_msgs::PointCloud2Modifier pcd_modifier(painted_pointcloud_msg);
  pcd_modifier.clear();
  pcd_modifier.reserve(tmp.width);
  painted_pointcloud_msg.width = tmp.width;
  constexpr int num_fields = 9;
  pcd_modifier.setPointCloud2Fields(
    num_fields, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
    sensor_msgs::msg::PointField::FLOAT32, "z", 1, sensor_msgs::msg::PointField::FLOAT32,
    "intensity", 1, sensor_msgs::msg::PointField::FLOAT32, "CAR", 1,
    sensor_msgs::msg::PointField::FLOAT32, "TRUCK", 1, sensor_msgs::msg::PointField::FLOAT32, "BUS",
    1, sensor_msgs::msg::PointField::FLOAT32, "BICYCLE", 1, sensor_msgs::msg::PointField::FLOAT32,
    "PEDESTRIAN", 1, sensor_msgs::msg::PointField::FLOAT32);
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

void PointPaintingFusionNode::fuseOnSingleImage(
  __attribute__((unused)) const sensor_msgs::msg::PointCloud2 & input_pointcloud_msg,
  __attribute__((unused)) const std::size_t image_id,
  const DetectedObjectsWithFeature & input_roi_msg,
  const sensor_msgs::msg::CameraInfo & camera_info,
  sensor_msgs::msg::PointCloud2 & painted_pointcloud_msg)
{
  auto num_bbox = (input_roi_msg.feature_objects).size();
  if (num_bbox == 0) {
    return;
  }

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

  // transform
  sensor_msgs::msg::PointCloud2 transformed_pointcloud;
  tf2::doTransform(painted_pointcloud_msg, transformed_pointcloud, transform_stamped);

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  const auto x_offset =
    transformed_pointcloud.fields.at(static_cast<size_t>(autoware_point_types::PointIndex::X))
      .offset;
  const auto y_offset =
    transformed_pointcloud.fields.at(static_cast<size_t>(autoware_point_types::PointIndex::Y))
      .offset;
  const auto z_offset =
    transformed_pointcloud.fields.at(static_cast<size_t>(autoware_point_types::PointIndex::Z))
      .offset;
  const auto car_offset = painted_pointcloud_msg.fields.at(4).offset;
  const auto trk_offset = painted_pointcloud_msg.fields.at(5).offset;
  const auto bus_offset = painted_pointcloud_msg.fields.at(6).offset;
  const auto bic_offset = painted_pointcloud_msg.fields.at(7).offset;
  const auto ped_offset = painted_pointcloud_msg.fields.at(8).offset;
  const auto p_step = transformed_pointcloud.point_step;
  // projection matrix
  Eigen::Matrix3f camera_projection;  // use only x,y,z
  camera_projection << camera_info.p.at(0), camera_info.p.at(1), camera_info.p.at(2),
    camera_info.p.at(4), camera_info.p.at(5), camera_info.p.at(6);
  /** dc : don't care

x    | f  x1 x2  dc ||xc|
y  = | y1  f y2  dc ||yc|
dc   | dc dc dc  dc ||zc|
                     |dc|
   **/

  auto objects = input_roi_msg.feature_objects;
  int iterations = transformed_pointcloud.data.size() / transformed_pointcloud.point_step;
  // iterate points
  // Requires 'OMP_NUM_THREADS=N'
#pragma omp parallel for
  for (int i = 0; i < iterations; i++) {
    int stride = p_step * i;
    unsigned char * data = &transformed_pointcloud.data[0];
    float p_x = *reinterpret_cast<const float *>(&data[stride + x_offset]);
    float p_y = *reinterpret_cast<const float *>(&data[stride + y_offset]);
    float p_z = *reinterpret_cast<const float *>(&data[stride + z_offset]);
    if (p_z <= 0.0 || (p_x / p_z) > tan_h_.at(image_id) || (p_x / p_z) < -tan_h_.at(image_id)) {
      continue;
    }
    // project
    Eigen::Vector3f normalized_projected_point = camera_projection * Eigen::Vector3f(p_x, p_y, p_z);
    // iterate 2d bbox
    for (const auto & feature_object : objects) {
      sensor_msgs::msg::RegionOfInterest roi = feature_object.feature.roi;
      // paint current point if it is inside bbox
      int label2d = feature_object.object.classification.front().label;
      if (
        !isUnknown(label2d) &&
        isInsideBbox(normalized_projected_point.x(), normalized_projected_point.y(), roi, p_z)) {
        data = &painted_pointcloud_msg.data[0];
        unsigned char * p_car = &data[stride + car_offset];
        unsigned char * p_trk = &data[stride + trk_offset];
        unsigned char * p_bus = &data[stride + bus_offset];
        unsigned char * p_bic = &data[stride + bic_offset];
        unsigned char * p_ped = &data[stride + ped_offset];
        *p_car = (isVehicle(label2d)) ? 1 : *p_car;
        *p_ped = (isPedestrian(label2d)) ? 1 : *p_ped;
        *p_bic = (isBicycle(label2d)) ? 1 : *p_bic;
        *p_trk = (isTruck(label2d)) ? 1 : *p_bic;
        *p_bus = (isBus(label2d)) ? 1 : *p_bic;
      }
#if 0
      //Parallelizing loop don't support push_back
      if (debugger_) {
	debug_image_points.push_back(normalized_projected_point);
      }
#endif
    }
  }
  double msec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                  std::chrono::system_clock::now() - start)
                  .count();
  std::cout << "Parallelism :" << omp_get_max_threads() << std::endl;
  std::cout << "#####Time for iterator of projection : " << msec << " msec" << std::endl;
  t_counter++;
  t_sum += msec;
  std::cout << "#####AVG time : " << t_sum / t_counter << " msec @" << t_counter << std::endl;

  for (const auto & feature_object : input_roi_msg.feature_objects) {
    debug_image_rois.push_back(feature_object.feature.roi);
  }

  if (debugger_) {
    debugger_->image_rois_ = debug_image_rois;
    debugger_->obstacle_points_ = debug_image_points;
    debugger_->publishImage(image_id, input_roi_msg.header.stamp);
  }
}

void PointPaintingFusionNode::postprocess(sensor_msgs::msg::PointCloud2 & painted_pointcloud_msg)
{
  std::vector<centerpoint::Box3D> det_boxes3d;
  bool is_success = detector_ptr_->detect(painted_pointcloud_msg, tf_buffer_, det_boxes3d);
  if (!is_success) {
    return;
  }

  autoware_auto_perception_msgs::msg::DetectedObjects output_obj_msg;
  output_obj_msg.header = painted_pointcloud_msg.header;
  for (const auto & box3d : det_boxes3d) {
    if (box3d.score < score_threshold_) {
      continue;
    }
    autoware_auto_perception_msgs::msg::DetectedObject obj;
    centerpoint::box3DToDetectedObject(box3d, class_names_, has_twist_, obj);
    output_obj_msg.objects.emplace_back(obj);
  }

  obj_pub_ptr_->publish(output_obj_msg);
}

bool PointPaintingFusionNode::out_of_scope(__attribute__((unused)) const DetectedObjects & obj)
{
  return false;
}
}  // namespace image_projection_based_fusion

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(image_projection_based_fusion::PointPaintingFusionNode)
