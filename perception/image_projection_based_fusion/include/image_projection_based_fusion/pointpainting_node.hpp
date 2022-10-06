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

#ifndef IMAGE_PROJECTION_BASED_FUSION__POINTPAINTING_NODE_HPP_
#define IMAGE_PROJECTION_BASED_FUSION__POINTPAINTING_NODE_HPP_

#include "image_projection_based_fusion/pointpainting_fusion/pointpainting_trt.hpp"

#include <image_projection_based_fusion/debugger.hpp>
#include <image_projection_based_fusion/utils/geometry.hpp>
#include <image_projection_based_fusion/utils/utils.hpp>
#include <lidar_centerpoint/centerpoint_trt.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/ros/debug_publisher.hpp>
#include <tier4_autoware_utils/system/stop_watch.hpp>

#include <autoware_auto_perception_msgs/msg/detected_objects.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <image_projection_based_fusion/utils/pp_approximate_time.h>
#include <message_filters/pass_through.h>
#include <message_filters/subscriber.h>
// #include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace image_projection_based_fusion
{
using autoware_auto_perception_msgs::msg::DetectedObject;
using autoware_auto_perception_msgs::msg::DetectedObjects;
using tier4_perception_msgs::msg::DetectedObjectsWithFeature;
using tier4_perception_msgs::msg::DetectedObjectWithFeature;

// template <class Msg, class ObjType>
class PointpaintingNode : public rclcpp::Node
{
public:
  explicit PointpaintingNode(const rclcpp::NodeOptions & options);

protected:
  void cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr input_camera_info_msg,
    const std::size_t camera_id);

  void fusionCallback(
    sensor_msgs::msg::PointCloud2::ConstSharedPtr input_msg,
    DetectedObjectsWithFeature::ConstSharedPtr input_roi0_msg, const std::size_t image_id);
  // DetectedObjectsWithFeature::ConstSharedPtr input_roi1_msg,
  // DetectedObjectsWithFeature::ConstSharedPtr input_roi2_msg);

  // void preprocess(const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_msg);

  void fuseOnSingleImage(
    // const sensor_msgs::msg::PointCloud2 & input_msg,
    const std::size_t image_id, const DetectedObjectsWithFeature & input_roi_msg,
    const sensor_msgs::msg::CameraInfo & camera_info, const double tan_h,
    uint64_t nanosec);  //, Msg & output_msg) = 0;

  // set args if you need
  void postprocess(sensor_msgs::msg::PointCloud2 & output_msg);

  // void publish(const Msg & output_msg);

  std::size_t rois_number_{1};
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // camera_info
  std::map<std::size_t, sensor_msgs::msg::CameraInfo> camera_info_map_;
  std::map<std::size_t, double> tan_h_map_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> camera_info_subs_;
  // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;

  // fusion
  // typename message_filters::Subscriber<Msg> sub_;
  // size_t paint_count_;
  size_t frame_count_;

  std::map<uint64_t, std::vector<bool>> is_painted_;
  std::map<uint64_t, std::chrono::time_point<std::chrono::steady_clock>> waittime_;
  // sensor_msgs::msg::PointCloud2 painted_pointcloud_msg_;
  std::map<uint64_t, sensor_msgs::msg::PointCloud2> painted_pointcloud_msgs_;
  // double one_sec_ =
  //   std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(1)).count();
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>
    painted_pointcloud_sub_;

  message_filters::PassThrough<DetectedObjectsWithFeature> passthrough_;
  // std::vector<rclcpp::Subscription<DetectedObjectsWithFeature>::SharedPtr> rois_subs_;
  std::vector<std::shared_ptr<message_filters::Subscriber<DetectedObjectsWithFeature>>> rois_subs_;
  inline void dummyCallback(DetectedObjectsWithFeature::ConstSharedPtr input)
  {
    passthrough_.add(input);
  }
  using SyncPolicy = message_filters::sync_policies::pp_ApproximateTime<
    sensor_msgs::msg::PointCloud2, DetectedObjectsWithFeature>;
  using Sync = message_filters::Synchronizer<SyncPolicy>;
  std::vector<std::shared_ptr<Sync>> sync_ptrs_;
  // typename std::shared_ptr<Sync> sync_ptr_1_;
  // typename std::shared_ptr<Sync> sync_ptr_2_;

  // output
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ptr_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_ptr_;
  rclcpp::Publisher<DetectedObjects>::SharedPtr obj_pub_ptr_;

  //   // debugger
  std::unique_ptr<tier4_autoware_utils::StopWatch<std::chrono::milliseconds>> stop_watch_ptr_{
    nullptr};
  std::unique_ptr<tier4_autoware_utils::DebugPublisher> debug_publisher_ptr_{nullptr};
  std::shared_ptr<Debugger> debugger_;
  //   virtual bool out_of_scope(const DetectedObjects & obj) = 0;
  //   float filter_scope_minx_;
  //   float filter_scope_maxx_;
  //   float filter_scope_miny_;
  //   float filter_scope_maxy_;
  //   float filter_scope_minz_;
  //   float filter_scope_maxz_;

  // detector
  float score_threshold_{0.0};
  std::vector<std::string> class_names_;
  std::vector<double> pointcloud_range;
  bool rename_car_to_truck_and_bus_{false};
  bool has_twist_{false};
  std::vector<int64_t> msg_offset_;

  std::unique_ptr<image_projection_based_fusion::PointPaintingTRT> detector_ptr_{nullptr};

  // bool out_of_scope(const DetectedObjects & obj);
};

}  // namespace image_projection_based_fusion

#endif  // IMAGE_PROJECTION_BASED_FUSION__POINTPAINTING_NODE_HPP_
