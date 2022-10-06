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

#define EIGEN_MPL2_ONLY

#include "image_projection_based_fusion/pointpainting_node.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <image_projection_based_fusion/utils/geometry.hpp>
#include <image_projection_based_fusion/utils/utils.hpp>
#include <lidar_centerpoint/centerpoint_config.hpp>
#include <lidar_centerpoint/preprocess/pointcloud_densification.hpp>
#include <lidar_centerpoint/ros_utils.hpp>
#include <lidar_centerpoint/utils.hpp>
#include <tier4_autoware_utils/geometry/geometry.hpp>
#include <tier4_autoware_utils/math/constants.hpp>

#include <tier4_perception_msgs/msg/detected_object_with_feature.hpp>

#include <boost/optional.hpp>

#include <fstream>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

// #include <typeinfo>
namespace image_projection_based_fusion
{
// template <class Msg, class ObjType>
PointpaintingNode::PointpaintingNode(const rclcpp::NodeOptions & options)
: Node("pointpainting", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
  // paint_count_ = 0;
  // frame_count_ = 0;
  rois_number_ = static_cast<std::size_t>(declare_parameter("rois_number", 1));
  if (rois_number_ < 1) {
    RCLCPP_WARN(
      this->get_logger(), "minimum rois_number is 1. current rois_number is %zu", rois_number_);
    rois_number_ = 1;
  }
  if (rois_number_ > 8) {
    RCLCPP_WARN(
      this->get_logger(), "maximum rois_number is 8. current rois_number is %zu", rois_number_);
    rois_number_ = 8;
  }
  // is_painted_.resize(rois_number_);

  // // subscribers
  // // sub_.subscribe(this, "input", rclcpp::QoS(1).get_rmw_qos_profile());

  // sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
  //   "~/input/pointcloud",
  //   rclcpp::SensorDataQoS{}.keep_last(1),  // rclcpp::QoS(1).get_rmw_qos_profile(),  //
  //   std::bind(&PointpaintingNode::preprocess, this, std::placeholders::_1));
  // pub_ptr_ =
  //   this->create_publisher<sensor_msgs::msg::PointCloud2>("initialized_pointcloud",
  //   rclcpp::QoS{1});
  debug_ptr_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("painted_pointcloud", rclcpp::QoS{1});
  obj_pub_ptr_ = this->create_publisher<DetectedObjects>("~/output/objects", rclcpp::QoS{1});

  camera_info_subs_.resize(rois_number_);
  for (std::size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    std::function<void(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)> fnc =
      std::bind(&PointpaintingNode::cameraInfoCallback, this, std::placeholders::_1, roi_i);
    camera_info_subs_.at(roi_i) = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "input/camera_info" + std::to_string(roi_i), rclcpp::QoS{1}.best_effort(), fnc);
  }

  rois_subs_.resize(rois_number_);
  for (std::size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    rois_subs_.at(roi_i) =
      std::make_shared<message_filters::Subscriber<DetectedObjectsWithFeature>>(
        this, "input/rois" + std::to_string(roi_i), rclcpp::QoS{1}.get_rmw_qos_profile());
  }

  // add dummy callback to enable passthrough filter
  rois_subs_.at(0)->registerCallback(
    std::bind(&PointpaintingNode::dummyCallback, this, std::placeholders::_1));

  // painted_pointcloud_sub_.subscribe(
  //   this, "~/input/pointcloud", rclcpp::QoS(1).get_rmw_qos_profile());
  // this, "initialized_pointcloud", rclcpp::QoS(1).get_rmw_qos_profile());
  painted_pointcloud_sub_ =
    std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
      this, "~/input/pointcloud",
      rmw_qos_profile_sensor_data);  // rclcpp::QoS{1}.get_rmw_qos_profile());

  msg_offset_ = this->declare_parameter<std::vector<int64_t>>("msg_offset_ns");
  // std::cout << msg_offset.at(1) << std::endl;

  sync_ptrs_.resize(rois_number_);
  for (std::size_t i = 0; i < rois_number_; ++i) {
    SyncPolicy policy(10);
    // policy.setMaxIntervalDuration(rclcpp::Duration(0, 50000000));
    std::vector<int64_t> offset{msg_offset_.at(0), msg_offset_.at(i + 1)};
    policy.setMsgOffset(offset);
    sync_ptrs_.at(i) = std::make_shared<Sync>(
      static_cast<const SyncPolicy &>(policy), *painted_pointcloud_sub_, *rois_subs_.at(i));
    sync_ptrs_.at(i)->registerCallback(std::bind(
      &PointpaintingNode::fusionCallback, this, std::placeholders::_1, std::placeholders::_2, i));
  }

  // sync_ptr_1_ = std::make_shared<Sync>(SyncPolicy(10), painted_pointcloud_sub_,
  // *rois_subs_.at(0)); sync_ptr_2_ = std::make_shared<Sync>(SyncPolicy(10),
  // painted_pointcloud_sub_, *rois_subs_.at(1)); sync_ptr_1_->registerCallback(std::bind(
  //   &PointpaintingNode::fusionCallback, this, std::placeholders::_1, std::placeholders::_2, 0));
  // sync_ptr_2_->registerCallback(std::bind(
  //   &PointpaintingNode::fusionCallback, this, std::placeholders::_1, std::placeholders::_2, 1));

  // switch (rois_number_) {
  //   case 1:
  //     sync_ptr_ = std::make_shared<Sync>(
  //       SyncPolicy(10), sub_, *rois_subs_.at(0), passthrough_, passthrough_, passthrough_,
  //       passthrough_, passthrough_, passthrough_, passthrough_);
  //     break;
  //   case 2:
  //     sync_ptr_ = std::make_shared<Sync>(
  //       SyncPolicy(10), sub_, *rois_subs_.at(0), *rois_subs_.at(1), passthrough_, passthrough_,
  //       passthrough_, passthrough_, passthrough_, passthrough_);
  //     break;
  //   case 3:
  //     sync_ptr_ = std::make_shared<Sync>(
  //       SyncPolicy(10), sub_, *rois_subs_.at(0), *rois_subs_.at(1), *rois_subs_.at(2),
  //       passthrough_, passthrough_, passthrough_, passthrough_, passthrough_);
  //     break;
  //   case 4:
  //     sync_ptr_ = std::make_shared<Sync>(
  //       SyncPolicy(10), sub_, *rois_subs_.at(0), *rois_subs_.at(1), *rois_subs_.at(2),
  //       *rois_subs_.at(3), passthrough_, passthrough_, passthrough_, passthrough_);
  //     break;
  //   case 5:
  //     sync_ptr_ = std::make_shared<Sync>(
  //       SyncPolicy(10), sub_, *rois_subs_.at(0), *rois_subs_.at(1), *rois_subs_.at(2),
  //       *rois_subs_.at(3), *rois_subs_.at(4), passthrough_, passthrough_, passthrough_);
  //     break;
  //   case 6:
  //     sync_ptr_ = std::make_shared<Sync>(
  //       SyncPolicy(10), sub_, *rois_subs_.at(0), *rois_subs_.at(1), *rois_subs_.at(2),
  //       *rois_subs_.at(3), *rois_subs_.at(4), *rois_subs_.at(5), passthrough_, passthrough_);
  //     break;
  //   case 7:
  //     sync_ptr_ = std::make_shared<Sync>(
  //       SyncPolicy(10), sub_, *rois_subs_.at(0), *rois_subs_.at(1), *rois_subs_.at(2),
  //       *rois_subs_.at(3), *rois_subs_.at(4), *rois_subs_.at(5), *rois_subs_.at(6),
  //       passthrough_);
  //     break;
  //   case 8:
  //     sync_ptr_ = std::make_shared<Sync>(
  //       SyncPolicy(10), sub_, *rois_subs_.at(0), *rois_subs_.at(1), *rois_subs_.at(2),
  //       *rois_subs_.at(3), *rois_subs_.at(4), *rois_subs_.at(5), *rois_subs_.at(6),
  //       *rois_subs_.at(7));
  //   default:
  //     return;
  // }

  // sync_ptr_->registerCallback(std::bind(
  //   &FusionNode::fusionCallback, this, std::placeholders::_1, std::placeholders::_2,
  //   std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
  //   std::placeholders::_7, std::placeholders::_8, std::placeholders::_9));

  // publisher
  // pub_ptr_ = this->create_publisher<Msg>("output", rclcpp::QoS{1});

  // debugger
  // if (declare_parameter("debug_mode", false)) {
  //   std::size_t image_buffer_size =
  //     static_cast<std::size_t>(declare_parameter("image_buffer_size", 15));
  //   debugger_ = std::make_shared<Debugger>(this, rois_number_, image_buffer_size);
  // }

  // filter_scope_minx_ = declare_parameter("filter_scope_minx", -100);
  // filter_scope_maxx_ = declare_parameter("filter_scope_maxx", 100);
  // filter_scope_miny_ = declare_parameter("filter_scope_miny", -100);
  // filter_scope_maxy_ = declare_parameter("filter_scope_maxy", 100);
  // filter_scope_minz_ = declare_parameter("filter_scope_minz", -100);
  // filter_scope_maxz_ = declare_parameter("filter_scope_maxz", 100);

  const float score_threshold =
    static_cast<float>(this->declare_parameter<double>("score_threshold", 0.4));
  const float circle_nms_dist_threshold =
    static_cast<float>(this->declare_parameter<double>("circle_nms_dist_threshold", 1.5));
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
  rename_car_to_truck_and_bus_ = this->declare_parameter("rename_car_to_truck_and_bus", false);
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

  centerpoint::NetworkParam encoder_param(encoder_onnx_path, encoder_engine_path, trt_precision);
  centerpoint::NetworkParam head_param(head_onnx_path, head_engine_path, trt_precision);
  centerpoint::DensificationParam densification_param(
    densification_world_frame_id, densification_num_past_frames);
  centerpoint::CenterPointConfig config(
    class_names_.size(), point_feature_size, max_voxel_size, pointcloud_range, voxel_size,
    downsample_factor, encoder_in_feature_size, score_threshold, circle_nms_dist_threshold);

  // create detector
  detector_ptr_ = std::make_unique<image_projection_based_fusion::PointPaintingTRT>(
    encoder_param, head_param, densification_param, config);

  // debugger
  if (declare_parameter("debug_mode", false)) {
    std::size_t image_buffer_size =
      static_cast<std::size_t>(declare_parameter("image_buffer_size", 15));
    debugger_ = std::make_shared<Debugger>(this, rois_number_, image_buffer_size);
  }

  // initialize debug tool
  {
    using tier4_autoware_utils::DebugPublisher;
    using tier4_autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ptr_ = std::make_unique<DebugPublisher>(this, "pointpainting");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }
}

// template <class Msg, class Obj>
void PointpaintingNode::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr input_camera_info_msg,
  const std::size_t camera_id)
{
  camera_info_map_[camera_id] = *input_camera_info_msg;

  // horizontal field of view
  auto fx = camera_info_map_[camera_id].k.at(0);
  auto x0 = camera_info_map_[camera_id].k.at(2);
  tan_h_map_[camera_id] = x0 / fx;
  // std::cout << "fx / x0:" << fx / x0 << std::endl;
  // std::cout << "x0 / fx" << x0 / fx << std::endl;
}

// template <class Msg, class Obj>
// void PointpaintingNode::preprocess(
//   const typename sensor_msgs::msg::PointCloud2::ConstSharedPtr input_msg __attribute__((unused)))
// {
//   // sensor_msgs::msg::PointCloud2 painted_pointcloud_msg;
//   // std::cout << "preprocess " << frame_count_ << std::endl;
//   painted_pointcloud_msg_ = *input_msg;

//   // set fields
//   sensor_msgs::PointCloud2Modifier pcd_modifier(painted_pointcloud_msg_);
//   pcd_modifier.clear();
//   painted_pointcloud_msg_.width = (*input_msg).width;
//   constexpr int num_fields = 7;
//   pcd_modifier.setPointCloud2Fields(
//     num_fields, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
//     sensor_msgs::msg::PointField::FLOAT32, "z", 1, sensor_msgs::msg::PointField::FLOAT32,
//     "intensity", 1, sensor_msgs::msg::PointField::FLOAT32, "CAR", 1,
//     sensor_msgs::msg::PointField::FLOAT32, "PEDESTRIAN", 1,
//     sensor_msgs::msg::PointField::FLOAT32, "BICYCLE", 1, sensor_msgs::msg::PointField::FLOAT32);
//   painted_pointcloud_msg_.point_step = num_fields * sizeof(float);

//   // filter points out of range
//   const auto painted_point_step = painted_pointcloud_msg_.point_step;
//   size_t j = 0;
//   sensor_msgs::PointCloud2Iterator<float> iter_painted_x(painted_pointcloud_msg_, "x");
//   sensor_msgs::PointCloud2Iterator<float> iter_painted_y(painted_pointcloud_msg_, "y");
//   sensor_msgs::PointCloud2Iterator<float> iter_painted_z(painted_pointcloud_msg_, "z");
//   for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(*input_msg, "x"),
//        iter_y(*input_msg, "y"), iter_z(*input_msg, "z");
//        iter_x != iter_x.end();
//        ++iter_x, ++iter_y, ++iter_z, ++iter_painted_x, ++iter_painted_y, ++iter_painted_z) {
//     if (
//       *iter_x <= pointcloud_range.at(0) || *iter_x >= pointcloud_range.at(3) ||
//       *iter_y <= pointcloud_range.at(1) || *iter_y >= pointcloud_range.at(4)) {
//       continue;
//     } else {
//       *iter_painted_x = *iter_x;
//       *iter_painted_y = *iter_y;
//       *iter_painted_z = *iter_z;
//       j += painted_point_step;
//     }
//   }
//   painted_pointcloud_msg_.data.resize(j);
//   painted_pointcloud_msg_.width = static_cast<uint32_t>(
//     painted_pointcloud_msg_.data.size() / painted_pointcloud_msg_.height /
//     painted_pointcloud_msg_.point_step);
//   painted_pointcloud_msg_.row_step =
//     static_cast<uint32_t>(painted_pointcloud_msg_.data.size() / painted_pointcloud_msg_.height);

//   pub_ptr_->publish(painted_pointcloud_msg_);
// }

// template <class Msg, class Obj>
void PointpaintingNode::fusionCallback(
  sensor_msgs::msg::PointCloud2::ConstSharedPtr input_msg,
  DetectedObjectsWithFeature::ConstSharedPtr input_roi0_msg, const std::size_t image_id)
{
  std::ofstream writing_file;
  std::string filename =
    "/home/tzhong/workspace/rosbag2/xx1_experiment/stamp_pp_node_map_offset.txt";
  writing_file.open(filename, std::ios::app);
  // std::cout << (*input_msg).header.stamp; builtin_interfaces::msg::Time
  // auto nanosec = std::make_pair((*input_msg).header.stamp.sec,(*input_msg).header.stamp.nanosec)
  uint64_t nanosec =
    (*input_msg).header.stamp.sec * (uint64_t)1e9 + (*input_msg).header.stamp.nanosec;
  // std::cout << std::fixed << std::setprecision(20)
  //           << ((*input_msg).header.stamp.sec * 1e9 + (*input_msg).header.stamp.nanosec)<<nanosec
  //           << std::endl;
  // int interval = ((*input_roi0_msg).header.stamp.sec - sec) * 1e9 +
  //                ((*input_roi0_msg).header.stamp.nanosec - nanosec);
  // std::cout << "interval:" << interval << std::endl;
  // if (interval > 6e8) {
  //   return;
  // }

  if (!painted_pointcloud_msgs_.count(nanosec)) {
    // std::cout << "preprocess " << frame_count_ << std::endl;
    // stop_watch_ptr_->toc("processing_time", true);
    painted_pointcloud_msgs_[nanosec] = *input_msg;
    std::vector<bool> tmp(rois_number_, false);
    is_painted_[nanosec] = tmp;
    waittime_[nanosec] = std::chrono::steady_clock::now();

    // set fields
    sensor_msgs::PointCloud2Modifier pcd_modifier(painted_pointcloud_msgs_[nanosec]);
    pcd_modifier.clear();
    painted_pointcloud_msgs_[nanosec].width = (*input_msg).width;
    constexpr int num_fields = 7;
    pcd_modifier.setPointCloud2Fields(
      num_fields, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
      sensor_msgs::msg::PointField::FLOAT32, "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "intensity", 1, sensor_msgs::msg::PointField::FLOAT32, "CAR", 1,
      sensor_msgs::msg::PointField::INT8, "PEDESTRIAN", 1, sensor_msgs::msg::PointField::INT8,
      "BICYCLE", 1, sensor_msgs::msg::PointField::INT8);
    painted_pointcloud_msgs_[nanosec].point_step = 4 * sizeof(float) + 3 * sizeof(int8_t);
    // num_fields * sizeof(float);

    // filter points out of range
    const auto painted_point_step = painted_pointcloud_msgs_[nanosec].point_step;
    size_t j = 0;
    sensor_msgs::PointCloud2Iterator<float> iter_painted_x(painted_pointcloud_msgs_[nanosec], "x");
    sensor_msgs::PointCloud2Iterator<float> iter_painted_y(painted_pointcloud_msgs_[nanosec], "y");
    sensor_msgs::PointCloud2Iterator<float> iter_painted_z(painted_pointcloud_msgs_[nanosec], "z");
    for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(*input_msg, "x"),
         iter_y(*input_msg, "y"), iter_z(*input_msg, "z");
         iter_x != iter_x.end();
         ++iter_x, ++iter_y, ++iter_z, ++iter_painted_x, ++iter_painted_y, ++iter_painted_z) {
      if (
        *iter_x <= pointcloud_range.at(0) || *iter_x >= pointcloud_range.at(3) ||
        *iter_y <= pointcloud_range.at(1) || *iter_y >= pointcloud_range.at(4) ||
        *iter_z <= pointcloud_range.at(2) || *iter_z >= pointcloud_range.at(5)) {
        continue;
      } else {
        *iter_painted_x = *iter_x;
        *iter_painted_y = *iter_y;
        *iter_painted_z = *iter_z;
        j += painted_point_step;
      }
    }
    painted_pointcloud_msgs_[nanosec].data.resize(j);
    painted_pointcloud_msgs_[nanosec].width = static_cast<uint32_t>(
      painted_pointcloud_msgs_[nanosec].data.size() / painted_pointcloud_msgs_[nanosec].height /
      painted_pointcloud_msgs_[nanosec].point_step);
    painted_pointcloud_msgs_[nanosec].row_step = static_cast<uint32_t>(
      painted_pointcloud_msgs_[nanosec].data.size() / painted_pointcloud_msgs_[nanosec].height);
  }

  auto offset_reset = (*input_roi0_msg).header.stamp.sec * (int64_t)1e9 +
                      (*input_roi0_msg).header.stamp.nanosec + msg_offset_.at(image_id + 1);

  std::string writing_text =
    "input_msg:" + std::to_string((*input_msg).header.stamp.sec) + " " +
    std::to_string((*input_msg).header.stamp.nanosec) + "| painted_pointcloud_msg_:" +
    std::to_string(painted_pointcloud_msgs_[nanosec].header.stamp.sec) + " " +
    std::to_string(painted_pointcloud_msgs_[nanosec].header.stamp.nanosec) + "\n " +
    std::to_string(image_id) + ":" + std::to_string(offset_reset) +
    "| interval: " + std::to_string(offset_reset - nanosec);
  // std::to_string((*input_roi0_msg).header.stamp.sec) + " " +
  // std::to_string((*input_roi0_msg).header.stamp.nanosec);
  writing_file << writing_text << std::endl;
  writing_file.close();

  // if (paint_count_ == 0) {
  stop_watch_ptr_->toc("processing_time", true);
  // }

  if (camera_info_map_.find(image_id) == camera_info_map_.end()) {
    RCLCPP_WARN(this->get_logger(), "no camera info. id is %zu", image_id);
    // continue;
  }
  if (debugger_) {
    debugger_->clear();
  }

  // if (paint_count_ == 0) {
  //   stop_watch_ptr_->toc("processing_time", true);
  // }
  // if (is_painted_.at(image_id) == false) {
  auto t_now = std::chrono::steady_clock::now();
  const auto one_sec =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(1)).count();
  double duration =
    static_cast<double>(
      (std::chrono::duration_cast<std::chrono::microseconds>(t_now - waittime_[nanosec])).count()) /
    one_sec;
  // std::chrono::milliseconds sec =
  //   std::chrono::duration_cast<std::chrono::milliseconds>(t_now - waittime_[nanosec]);
  // double duration = static_cast<double>(sec.count());
  // // std::time_t waittime_[nanosec] = std::chrono::system_clock::to_time_t(waittime_[nanosec]);
  // // std::cout << std::ctime(&std::chrono::system_clock::to_time_t(waittime_[nanosec])) <<
  // // std::endl;
  // std::cout << std::fixed << std::setprecision(20) << nanosec << "---" << image_id << ":"
  //           << std::ctime(&std::chrono::system_clock::to_time_t(waittime_[nanosec])) << "|"
  //           << duration << "ms" << std::endl;
  // std::cout << nanosec << " | duration:"
  //           << duration
  //           //           // << static_cast<double>(
  //           //           // std::chrono::duration_cast<std::chrono::microseconds>(t_now).count())
  //           << std::endl;

  // if (duration > 0.11) {
  //   std::cout << "discard" << std::endl;
  //   debug_ptr_->publish(painted_pointcloud_msgs_[nanosec]);
  //   postprocess(painted_pointcloud_msgs_[nanosec]);
  //   is_painted_.erase(nanosec);
  //   painted_pointcloud_msgs_.erase(nanosec);
  //   waittime_.erase(nanosec);

  //   return;
  // }

  fuseOnSingleImage(
    image_id, *input_roi0_msg, camera_info_map_.at(image_id), tan_h_map_.at(image_id), nanosec);
  is_painted_[nanosec].at(image_id) = true;

  // double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
  // if (paint_count_ == rois_number_) {
  // if (!std::count(is_painted_.begin(), is_painted_.end(), false)) {
  if (
    !std::count(is_painted_[nanosec].begin(), is_painted_[nanosec].end(), false) ||
    duration > 0.10) {
    // writing_text = "\n";
    // writing_file << writing_text << std::endl;
    // writing_file.close();
    // ++frame_count_;
    // std::cout << frame_count_ << std::endl;
    // paint_count_ = 0;
    // std::fill(is_painted_.begin(), is_painted_.end(), false);

    // std::cout << "pub" << std::endl;
    // if (stop_watch_ptr_) {
    // double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
    // std::cout << "processing_time_ms:" << processing_time_ms;
    // }
    debug_ptr_->publish(painted_pointcloud_msgs_[nanosec]);

    postprocess(painted_pointcloud_msgs_[nanosec]);

    auto iter_p = painted_pointcloud_msgs_.begin();
    auto iter_w = waittime_.begin();
    for (auto iter = is_painted_.begin(); iter != is_painted_.end(); ++iter, ++iter_p, ++iter_w) {
      if (iter->first == nanosec) {
        is_painted_.erase(is_painted_.begin(), iter);
        painted_pointcloud_msgs_.erase(painted_pointcloud_msgs_.begin(), iter_p);
        waittime_.erase(waittime_.begin(), iter_w);
        break;
      }
    }
    is_painted_.erase(nanosec);
    painted_pointcloud_msgs_.erase(nanosec);
    waittime_.erase(nanosec);

    if (debug_publisher_ptr_ && stop_watch_ptr_) {
      // double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
      double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
      // std::cout << "|| cyclic_time:" << cyclic_time_ms << std::endl;
      debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
        "debug/cyclic_time_ms", cyclic_time_ms);
      // debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      //   "debug/processing_time_ms", processing_time_ms);
    }
  }
}

void PointpaintingNode::fuseOnSingleImage(
  // const sensor_msgs::msg::PointCloud2 & painted_pointcloud_msg,
  const std::size_t image_id, const DetectedObjectsWithFeature & input_roi_msg,
  const sensor_msgs::msg::CameraInfo & camera_info, const double tan_h, uint64_t nanosec)
// sensor_msgs::msg::PointCloud2 & painted_pointcloud_msg)
{
  // std::cout << tan_h << std::endl;
  // std::cout << image_id << "_painted_pointcloud_msg_:" <<
  // painted_pointcloud_msg_.header.stamp.sec
  //           << " " << painted_pointcloud_msg_.header.stamp.nanosec << std::endl;
  std::vector<sensor_msgs::msg::RegionOfInterest> debug_image_rois;
  std::vector<Eigen::Vector2d> debug_image_points;

  // get transform from cluster frame id to camera optical frame id
  geometry_msgs::msg::TransformStamped transform_stamped;
  {
    const auto transform_stamped_optional = getTransformStamped(
      tf_buffer_, /*target*/ camera_info.header.frame_id,
      /*source*/ painted_pointcloud_msgs_[nanosec].header.frame_id, camera_info.header.stamp);
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
  tf2::doTransform(painted_pointcloud_msgs_[nanosec], transformed_pointcloud, transform_stamped);

  // iterate points
  sensor_msgs::PointCloud2Iterator<float> iter_painted_intensity(
    painted_pointcloud_msgs_[nanosec], "intensity");
  sensor_msgs::PointCloud2Iterator<int8_t> iter_car(painted_pointcloud_msgs_[nanosec], "CAR");
  sensor_msgs::PointCloud2Iterator<int8_t> iter_ped(
    painted_pointcloud_msgs_[nanosec], "PEDESTRIAN");
  sensor_msgs::PointCloud2Iterator<int8_t> iter_bic(painted_pointcloud_msgs_[nanosec], "BICYCLE");
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(transformed_pointcloud, "x"),
       iter_y(transformed_pointcloud, "y"), iter_z(transformed_pointcloud, "z");
       iter_x != iter_x.end();
       ++iter_x, ++iter_y, ++iter_z, ++iter_painted_intensity, ++iter_car, ++iter_ped, ++iter_bic) {
    // filter points
    if (*iter_z <= 0.0 || (*iter_x / *iter_z) > tan_h || (*iter_x / *iter_z) < (-tan_h)) {
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
        // *iter_car = 1;
        switch (feature_object.object.classification.front().label) {
          case autoware_auto_perception_msgs::msg::ObjectClassification::CAR:
            *iter_car = 1;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::TRUCK:
            *iter_car = 1;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::TRAILER:
            *iter_car = 1;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::BUS:
            *iter_car = 1;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::PEDESTRIAN:
            *iter_ped = 1;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::BICYCLE:
            *iter_bic = 1;
            break;
          case autoware_auto_perception_msgs::msg::ObjectClassification::MOTORCYCLE:
            *iter_bic = 1;
            break;
        }
        if (debugger_) {
          debug_image_points.push_back(normalized_projected_point);
        }
      }
    }
  }

  for (const auto & feature_object : input_roi_msg.feature_objects) {
    debug_image_rois.push_back(feature_object.feature.roi);
  }

  if (debugger_) {
    debugger_->image_rois_ = debug_image_rois;
    debugger_->obstacle_points_ = debug_image_points;
    debugger_->publishImage(image_id, input_roi_msg.header.stamp);
  }
}

// template <class Msg, class Obj>
void PointpaintingNode::postprocess(sensor_msgs::msg::PointCloud2 & painted_pointcloud_msg
                                    __attribute__((unused)))
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
    centerpoint::box3DToDetectedObject(
      box3d, class_names_, rename_car_to_truck_and_bus_, has_twist_, obj);
    output_obj_msg.objects.emplace_back(obj);
  }

  obj_pub_ptr_->publish(output_obj_msg);
}

// template <class Msg, class Obj>
// void PointpaintingNode::publish(const sensor_msgs::msg::PointCloud2 & output_msg)
// {
//   if (pub_ptr_->get_subscription_count() < 1) {
//     return;
//   }
//   pub_ptr_->publish(output_msg);
// }

// template class FusionNode<DetectedObjects, DetectedObject>;
// template class FusionNode<DetectedObjectsWithFeature, DetectedObjectWithFeature>;
// template class FusionNode<sensor_msgs::msg::PointCloud2, DetectedObjects>;
}  // namespace image_projection_based_fusion

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(image_projection_based_fusion::PointpaintingNode)