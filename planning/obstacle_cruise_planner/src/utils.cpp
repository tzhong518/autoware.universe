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

#include "obstacle_cruise_planner/utils.hpp"

namespace obstacle_cruise_utils
{
bool isVehicle(const uint8_t label)
{
  return label == ObjectClassification::CAR || label == ObjectClassification::TRUCK ||
         label == ObjectClassification::BUS || label == ObjectClassification::MOTORCYCLE;
}

visualization_msgs::msg::Marker getObjectMarker(
  const geometry_msgs::msg::Pose & obstacle_pose, size_t idx, const std::string & ns,
  const double r, const double g, const double b)
{
  const auto current_time = rclcpp::Clock().now();

  auto marker = tier4_autoware_utils::createDefaultMarker(
    "map", current_time, ns, idx, visualization_msgs::msg::Marker::SPHERE,
    tier4_autoware_utils::createMarkerScale(2.0, 2.0, 2.0),
    tier4_autoware_utils::createMarkerColor(r, g, b, 0.8));

  marker.pose = obstacle_pose;

  return marker;
}

boost::optional<geometry_msgs::msg::Pose> calcForwardPose(
  const autoware_auto_planning_msgs::msg::Trajectory & traj, const size_t start_idx,
  const double target_length)
{
  if (traj.points.empty()) {
    return {};
  }

  size_t search_idx = start_idx;
  double length_to_search_idx = 0.0;
  for (; search_idx < traj.points.size(); ++search_idx) {
    length_to_search_idx = motion_utils::calcSignedArcLength(traj.points, start_idx, search_idx);
    if (length_to_search_idx > target_length) {
      break;
    } else if (search_idx == traj.points.size() - 1) {
      return {};
    }
  }

  if (search_idx == 0 && !traj.points.empty()) {
    return traj.points.at(0).pose;
  }

  const auto & pre_pose = traj.points.at(search_idx - 1).pose;
  const auto & next_pose = traj.points.at(search_idx).pose;

  geometry_msgs::msg::Pose target_pose;

  // lerp position
  const double seg_length =
    tier4_autoware_utils::calcDistance2d(pre_pose.position, next_pose.position);
  const double lerp_ratio = (length_to_search_idx - target_length) / seg_length;

  return tier4_autoware_utils::calcInterpolatedPose(pre_pose, next_pose, lerp_ratio);
}

boost::optional<geometry_msgs::msg::Pose> lerpByTimeStamp(
  const autoware_auto_perception_msgs::msg::PredictedPath & path, const rclcpp::Duration & rel_time)
{
  auto clock{rclcpp::Clock{RCL_ROS_TIME}};
  if (
    path.path.empty() || rel_time < rclcpp::Duration::from_seconds(0.0) ||
    rel_time > rclcpp::Duration(path.time_step) * (static_cast<double>(path.path.size()) - 1)) {
    return boost::none;
  }

  for (size_t i = 1; i < path.path.size(); ++i) {
    const auto & pt = path.path.at(i);
    const auto & prev_pt = path.path.at(i - 1);
    if (rel_time <= rclcpp::Duration(path.time_step) * static_cast<double>(i)) {
      const auto offset = rel_time - rclcpp::Duration(path.time_step) * static_cast<double>(i - 1);
      const auto ratio = offset.seconds() / rclcpp::Duration(path.time_step).seconds();
      return tier4_autoware_utils::calcInterpolatedPose(prev_pt, pt, ratio);
    }
  }

  return boost::none;
}

boost::optional<geometry_msgs::msg::Pose> getCurrentObjectPoseFromPredictedPath(
  const autoware_auto_perception_msgs::msg::PredictedPath & predicted_path,
  const rclcpp::Time & obj_base_time, const rclcpp::Time & current_time)
{
  const auto rel_time = current_time - obj_base_time;
  if (rel_time.seconds() < 0.0) {
    return boost::none;
  }

  return lerpByTimeStamp(predicted_path, rel_time);
}

boost::optional<geometry_msgs::msg::Pose> getCurrentObjectPoseFromPredictedPaths(
  const std::vector<autoware_auto_perception_msgs::msg::PredictedPath> & predicted_paths,
  const rclcpp::Time & obj_base_time, const rclcpp::Time & current_time)
{
  if (predicted_paths.empty()) {
    return boost::none;
  }
  // Get the most reliable path
  const auto predicted_path = std::max_element(
    predicted_paths.begin(), predicted_paths.end(),
    [](
      const autoware_auto_perception_msgs::msg::PredictedPath & a,
      const autoware_auto_perception_msgs::msg::PredictedPath & b) {
      return a.confidence < b.confidence;
    });

  return getCurrentObjectPoseFromPredictedPath(*predicted_path, obj_base_time, current_time);
}

geometry_msgs::msg::Pose getCurrentObjectPose(
  const autoware_auto_perception_msgs::msg::PredictedObject & predicted_object,
  const rclcpp::Time & obj_base_time, const rclcpp::Time & current_time, const bool use_prediction)
{
  if (!use_prediction) {
    return predicted_object.kinematics.initial_pose_with_covariance.pose;
  }

  std::vector<autoware_auto_perception_msgs::msg::PredictedPath> predicted_paths;
  for (const auto & path : predicted_object.kinematics.predicted_paths) {
    predicted_paths.push_back(path);
  }
  const auto interpolated_pose =
    getCurrentObjectPoseFromPredictedPaths(predicted_paths, obj_base_time, current_time);

  if (!interpolated_pose) {
    RCLCPP_WARN(
      rclcpp::get_logger("ObstacleCruisePlanner"), "Failed to find the interpolated obstacle pose");
    return predicted_object.kinematics.initial_pose_with_covariance.pose;
  }

  return interpolated_pose.get();
}

boost::optional<TargetObstacle> getClosestStopObstacle(
  const autoware_auto_planning_msgs::msg::Trajectory & traj,
  const std::vector<TargetObstacle> & target_obstacles)
{
  if (target_obstacles.empty()) {
    return boost::none;
  }

  boost::optional<TargetObstacle> closest_stop_obstacle = boost::none;
  double dist_to_closest_stop_obstacle = std::numeric_limits<double>::max();
  for (const auto & obstacle : target_obstacles) {
    // Ignore obstacle that has not stopped
    if (!obstacle.has_stopped) {
      continue;
    }

    const double dist_to_stop_obstacle =
      motion_utils::calcSignedArcLength(traj.points, 0, obstacle.collision_point);
    if (dist_to_stop_obstacle < dist_to_closest_stop_obstacle) {
      dist_to_closest_stop_obstacle = dist_to_stop_obstacle;
      closest_stop_obstacle = obstacle;
    }
  }
  return closest_stop_obstacle;
}

std::string toHexString(const unique_identifier_msgs::msg::UUID & id)
{
  std::stringstream ss;
  for (auto i = 0; i < 16; ++i) {
    ss << std::hex << std::setfill('0') << std::setw(2) << +id.uuid[i];
  }
  return ss.str();
}
}  // namespace obstacle_cruise_utils
