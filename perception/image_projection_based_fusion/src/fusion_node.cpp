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

#include "image_projection_based_fusion/fusion_node.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tier4_perception_msgs/msg/detected_object_with_feature.hpp>

#include <boost/optional.hpp>

#include <cmath>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <typeinfo>
namespace image_projection_based_fusion
{

template <class Msg, class ObjType>
FusionNode<Msg, ObjType>::FusionNode(
  const std::string & node_name, const rclcpp::NodeOptions & options)
: Node(node_name, options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
  // set rois_number
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

  // Set parameters
  match_threshold_ms_ = static_cast<int>(declare_parameter("match_threshold_ms", 50));
  timeout_sec_ = static_cast<double>(declare_parameter("timeout_sec", 0.1));
  input_offset_ms_ = declare_parameter("input_offset_ms", std::vector<int64_t>{});
  if (!input_offset_ms_.empty() && rois_number_ != input_offset_ms_.size()) {
    RCLCPP_ERROR(get_logger(), "The number of topics does not match the number of offsets.");
    return;
  }

  // sub camera info
  camera_info_subs_.resize(rois_number_);
  for (std::size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    std::function<void(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)> fnc =
      std::bind(&FusionNode::cameraInfoCallback, this, std::placeholders::_1, roi_i);
    camera_info_subs_.at(roi_i) = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "input/camera_info" + std::to_string(roi_i), rclcpp::QoS{1}.best_effort(), fnc);
  }

  // sub rois
  rois_subs_.resize(rois_number_);
  for (std::size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    std::function<void(const DetectedObjectsWithFeature::ConstSharedPtr msg)> roi_callback =
      std::bind(&FusionNode::roiCallback, this, std::placeholders::_1, roi_i);
    // rois_subs_.at(roi_i) =
    //   std::make_shared<message_filters::Subscriber<DetectedObjectsWithFeature>>(
    //     this, "input/rois" + std::to_string(roi_i), rclcpp::QoS{1}.get_rmw_qos_profile());
    rois_subs_.at(roi_i) = this->create_subscription<DetectedObjectsWithFeature>(
      "input/rois" + std::to_string(roi_i), rclcpp::QoS{1}.best_effort(), roi_callback);
  }

  // subscribers
  std::function<void(const typename Msg::ConstSharedPtr msg)> sub_callback =
    std::bind(&FusionNode::subCallback, this, std::placeholders::_1);
  sub_ = this->create_subscription<Msg>("input", rclcpp::QoS(1).best_effort(), sub_callback);

  // publisher
  pub_ptr_ = this->create_publisher<Msg>("output", rclcpp::QoS{1});

  // Set timer
  {
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(timeout_sec_));
    std::cout << period_ns.count() << "|" << match_threshold_ms_ << std::endl;
    timer_ = rclcpp::create_timer(
      this, get_clock(), period_ns, std::bind(&FusionNode::timer_callback, this));
  }

  // debugger
  //   if (declare_parameter("debug_mode", false)) {
  //     std::size_t image_buffer_size =
  //       static_cast<std::size_t>(declare_parameter("image_buffer_size", 15));
  //     debugger_ = std::make_shared<Debugger>(this, rois_number_, image_buffer_size);
  //   }
  // initialize debug tool
  {
    using tier4_autoware_utils::DebugPublisher;
    using tier4_autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ = std::make_unique<DebugPublisher>(this, "concatenate_data_synchronizer");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }

  //   filter_scope_minx_ = declare_parameter("filter_scope_minx", -100);
  //   filter_scope_maxx_ = declare_parameter("filter_scope_maxx", 100);
  //   filter_scope_miny_ = declare_parameter("filter_scope_miny", -100);
  //   filter_scope_maxy_ = declare_parameter("filter_scope_maxy", 100);
  //   filter_scope_minz_ = declare_parameter("filter_scope_minz", -100);
  //   filter_scope_maxz_ = declare_parameter("filter_scope_maxz", 100);
}

template <class Msg, class Obj>
void FusionNode<Msg, Obj>::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr input_camera_info_msg,
  const std::size_t camera_id)
{
  camera_info_map_[camera_id] = *input_camera_info_msg;
}

template <class Msg, class Obj>
void FusionNode<Msg, Obj>::preprocess(Msg & ouput_msg __attribute__((unused)))
{
  // do nothing by default
}

template <class Msg, class Obj>
void FusionNode<Msg, Obj>::subCallback(const typename Msg::ConstSharedPtr input_msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(timeout_sec_));
  try {
    setPeriod(period.count());
  } catch (rclcpp::exceptions::RCLError & ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", ex.what());
  }
  timer_->reset();

  output_msg_ = std::make_shared<Msg>(*input_msg);

  preprocess(*output_msg_);

  int64_t nanosec =
    (*output_msg_).header.stamp.sec * (int64_t)1e9 + (*output_msg_).header.stamp.nanosec;

  // if matching rois exist, fuseonsingle(output_msg)
  std::vector<bool> is_painted(rois_number_, false);
  is_waiting_ = true;
  std::cout << "----------\n"
            << "sub nanosec:" << nanosec << "|"
            << std::count(is_painted.begin(), is_painted.end(), true) << std::endl;
  while (is_waiting_ == true) {
    int64_t min_interval = 1e9;
    int64_t matched_stamp;
    std::size_t roi_i;
    if (roi_stdmap_.size() > 0) {
      for (const auto & [k, v] : roi_stdmap_) {
        // auto iter = roi_stdmap_.end();
        int64_t newstamp = nanosec + input_offset_ms_.at(v.first) * (int64_t)1e6;
        int64_t interval = abs(int64_t(k) - newstamp);
        // std::cout << k << " - " << newstamp << " = " << interval << std::endl;
        if (interval < min_interval && interval < match_threshold_ms_ * (int64_t)1e6) {
          min_interval = interval;
          matched_stamp = k;
          roi_i = v.first;
          //     //   std::cout << "matched:" << min_interval << "|" << matched_stamp << std::endl;
          //   }

          // if (std::count(is_painted.begin(), is_painted.end(), true) == int(rois_number_)) {
          //   break;
          // }
        }
      }
      // fuseonSingle
      fuseOnSingleImage(
        *input_msg, roi_i, *(roi_stdmap_[matched_stamp].second), camera_info_map_.at(roi_i),
        *output_msg_);

      roi_stdmap_.erase(matched_stamp);
      is_painted.at(roi_i) = true;
      std::cout << "roi_stdmap_.size():" << roi_stdmap_.size() << " | "
                << std::count(is_painted.begin(), is_painted.end(), true) << " | " << roi_i
                << "| matched_stamp:" << matched_stamp << "| min_interval:" << min_interval
                << std::endl;
      if (std::count(is_painted.begin(), is_painted.end(), true) == (int(rois_number_))) {
        is_waiting_ = false;
        std::cout << "all subscribed" << std::endl;
        timer_->cancel();
        postprocess();
        publish(*output_msg_);
        break;
      }
    }
  }
}

template <class Msg, class Obj>
void FusionNode<Msg, Obj>::roiCallback(
  const DetectedObjectsWithFeature::ConstSharedPtr input_roi_msg, const std::size_t roi_i)
{
  int64_t nanosec =
    (*input_roi_msg).header.stamp.sec * (int64_t)1e9 + (*input_roi_msg).header.stamp.nanosec;
  //   std::cout << roi_i << ":" << nanosec << std::endl;
  auto cache_roi_msg = std::make_pair<int, DetectedObjectsWithFeature::SharedPtr>(
    roi_i, std::make_shared<DetectedObjectsWithFeature>(*input_roi_msg));
  roi_stdmap_[nanosec] = cache_roi_msg;
}

template <class Msg, class Obj>
void FusionNode<Msg, Obj>::postprocess()
{
  // timer_->cancel();
  std::cout << "postprocess" << std::endl;
  // do nothing by default
}

template <class Msg, class Obj>
void FusionNode<Msg, Obj>::timer_callback()
{
  std::cout << "timeout" << std::endl;
  using std::chrono_literals::operator""ms;
  timer_->cancel();
  if (mutex_.try_lock()) {
    is_waiting_ = false;
    postprocess();
    // publish(output_msg);
    mutex_.unlock();
  } else {
    try {
      std::chrono::nanoseconds period = 10ms;
      setPeriod(period.count());
    } catch (rclcpp::exceptions::RCLError & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", ex.what());
    }
    timer_->reset();
  }
}

template <class Msg, class Obj>
void FusionNode<Msg, Obj>::setPeriod(const int64_t new_period)
{
  if (!timer_) {
    return;
  }
  int64_t old_period = 0;
  rcl_ret_t ret = rcl_timer_get_period(timer_->get_timer_handle().get(), &old_period);
  if (ret != RCL_RET_OK) {
    rclcpp::exceptions::throw_from_rcl_error(ret, "Couldn't get old period");
  }
  ret = rcl_timer_exchange_period(timer_->get_timer_handle().get(), new_period, &old_period);
  if (ret != RCL_RET_OK) {
    rclcpp::exceptions::throw_from_rcl_error(ret, "Couldn't exchange_period");
  }
}

template <class Msg, class Obj>
void FusionNode<Msg, Obj>::publish(const Msg & output_msg)
{
  if (pub_ptr_->get_subscription_count() < 1) {
    return;
  }
  pub_ptr_->publish(output_msg);
}

template class FusionNode<DetectedObjects, DetectedObject>;
template class FusionNode<DetectedObjectsWithFeature, DetectedObjectWithFeature>;
template class FusionNode<sensor_msgs::msg::PointCloud2, DetectedObjects>;
}  // namespace image_projection_based_fusion