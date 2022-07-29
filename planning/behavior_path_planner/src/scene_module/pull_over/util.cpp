// Copyright 2021 Tier IV, Inc.
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

#include "behavior_path_planner/scene_module/pull_over/util.hpp"

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/utils/path_shifter.hpp"

#include <lanelet2_extension/utility/query.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/geometry/boost_geometry.hpp>

#include <boost/geometry/algorithms/dispatch/distance.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <tf2/utils.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using tier4_autoware_utils::calcOffsetPose;
using tier4_autoware_utils::inverseTransformPoint;

namespace behavior_path_planner
{
namespace pull_over_utils
{
PathWithLaneId combineReferencePath(const PathWithLaneId path1, const PathWithLaneId path2)
{
  PathWithLaneId path;
  path.points.insert(path.points.end(), path1.points.begin(), path1.points.end());

  // skip overlapping point
  path.points.insert(path.points.end(), next(path2.points.begin()), path2.points.end());

  return path;
}

bool isPathInLanelets(
  const PathWithLaneId & path, const lanelet::ConstLanelets & original_lanelets,
  const lanelet::ConstLanelets & target_lanelets)
{
  for (const auto & pt : path.points) {
    bool is_in_lanelet = false;
    for (const auto & llt : original_lanelets) {
      if (lanelet::utils::isInLanelet(pt.point.pose, llt, 0.1)) {
        is_in_lanelet = true;
      }
    }
    for (const auto & llt : target_lanelets) {
      if (lanelet::utils::isInLanelet(pt.point.pose, llt, 0.1)) {
        is_in_lanelet = true;
      }
    }
    if (!is_in_lanelet) {
      return false;
    }
  }
  return true;
}

std::vector<ShiftParkingPath> getShiftParkingPaths(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & original_lanelets,
  const lanelet::ConstLanelets & target_lanelets, const Pose & current_pose, const Pose & goal_pose,
  [[maybe_unused]] const Twist & twist, const BehaviorPathPlannerParameters & common_parameter,
  const PullOverParameters & parameter)
{
  std::vector<ShiftParkingPath> candidate_paths;

  if (original_lanelets.empty() || target_lanelets.empty()) {
    return candidate_paths;
  }

  // rename parameter
  const double backward_path_length = common_parameter.backward_path_length;
  const double pull_over_velocity = parameter.pull_over_velocity;
  const double after_pull_over_straight_distance = parameter.after_pull_over_straight_distance;
  const double margin = parameter.margin_from_boundary;
  const double minimum_lateral_jerk = parameter.minimum_lateral_jerk;
  const double maximum_lateral_jerk = parameter.maximum_lateral_jerk;
  const double deceleration_interval = parameter.deceleration_interval;
  const int pull_over_sampling_num = parameter.pull_over_sampling_num;
  const double jerk_resolution =
    std::abs(maximum_lateral_jerk - minimum_lateral_jerk) / pull_over_sampling_num;

  double distance_to_shoulder_lane_boundary =
    util::getDistanceToShoulderBoundary(route_handler.getShoulderLanelets(), current_pose);
  double offset_from_current_pose =
    distance_to_shoulder_lane_boundary + common_parameter.vehicle_width / 2 + margin;

  for (double lateral_jerk = minimum_lateral_jerk; lateral_jerk <= maximum_lateral_jerk;
       lateral_jerk += jerk_resolution) {
    PathShifter path_shifter;
    ShiftedPath shifted_path;
    ShiftParkingPath candidate_path;

    double pull_over_distance = path_shifter.calcLongitudinalDistFromJerk(
      abs(offset_from_current_pose), lateral_jerk, pull_over_velocity);

    // calculate straight distance before pull over
    double straight_distance;
    {
      const auto arc_position_goal =
        lanelet::utils::getArcCoordinates(original_lanelets, goal_pose);
      const auto arc_position_pose =
        lanelet::utils::getArcCoordinates(original_lanelets, current_pose);
      straight_distance = arc_position_goal.length - after_pull_over_straight_distance -
                          pull_over_distance - arc_position_pose.length;
    }

    // shift end point in shoulder lane
    const auto shift_end_point = [&]() {
      const auto arc_position_goal = lanelet::utils::getArcCoordinates(target_lanelets, goal_pose);
      const double s_start = arc_position_goal.length - after_pull_over_straight_distance;
      const double s_end = s_start + std::numeric_limits<double>::epsilon();
      const auto path = route_handler.getCenterLinePath(target_lanelets, s_start, s_end, true);
      return path.points.front();
    }();

    PathWithLaneId road_lane_reference_path;
    {
      const auto arc_position = lanelet::utils::getArcCoordinates(original_lanelets, current_pose);
      const auto arc_position_ref2_front =
        lanelet::utils::getArcCoordinates(original_lanelets, shift_end_point.point.pose);
      const double s_start = arc_position.length - backward_path_length;
      const double s_end = arc_position_ref2_front.length - pull_over_distance;
      road_lane_reference_path = route_handler.getCenterLinePath(original_lanelets, s_start, s_end);
      // decelerate velocity linearly to minimum pull over velocity
      // ( or accelerate if original velocity is lower than minimum velocity )
      for (auto & point : road_lane_reference_path.points) {
        const auto arclength =
          lanelet::utils::getArcCoordinates(original_lanelets, point.point.pose).length;
        const double distance_to_pull_over_start = std::max(0.0, s_end - arclength);
        point.point.longitudinal_velocity_mps = std::min(
          point.point.longitudinal_velocity_mps,
          static_cast<float>(
            (distance_to_pull_over_start / deceleration_interval) *
              (point.point.longitudinal_velocity_mps - pull_over_velocity) +
            pull_over_velocity));
      }
    }

    if (road_lane_reference_path.points.empty()) {
      RCLCPP_ERROR_STREAM(
        rclcpp::get_logger("behavior_path_planner").get_child("pull_over").get_child("util"),
        "reference path is empty!! something wrong...");
      continue;
    }

    PathWithLaneId target_lane_reference_path;
    {
      const lanelet::ArcCoordinates pull_over_start_arc_position =
        lanelet::utils::getArcCoordinates(
          target_lanelets, road_lane_reference_path.points.back().point.pose);
      const double s_start = pull_over_start_arc_position.length;
      const auto arc_position_goal = lanelet::utils::getArcCoordinates(target_lanelets, goal_pose);
      const double s_end = arc_position_goal.length;
      target_lane_reference_path = route_handler.getCenterLinePath(target_lanelets, s_start, s_end);
      // distance between shoulder lane's left boundary and shoulder lane center
      double distance_shoulder_to_left_bound = util::getDistanceToShoulderBoundary(
        route_handler.getShoulderLanelets(), shift_end_point.point.pose);

      // distance between shoulder lane center and target line
      double distance_shoulder_to_target =
        distance_shoulder_to_left_bound + common_parameter.vehicle_width / 2 + margin;

      // Apply shifting shoulder lane to adjust to target line
      double offset = -distance_shoulder_to_target;
      for (size_t i = 0; i < target_lane_reference_path.points.size(); ++i) {
        {
          if (fabs(offset) < 1.0e-8) {
            RCLCPP_WARN_STREAM(
              rclcpp::get_logger("behavior_path_planner").get_child("pull_over").get_child("util"),
              "no offset from current lane center.");
          }

          auto & p = target_lane_reference_path.points.at(i).point.pose;
          double yaw = tf2::getYaw(p.orientation);
          p.position.x -= std::sin(yaw) * offset;
          p.position.y += std::cos(yaw) * offset;
        }
        path_shifter.setPath(target_lane_reference_path);
      }
    }
    ShiftPoint shift_point;
    {
      shift_point.start = road_lane_reference_path.points.back().point.pose;
      shift_point.end = shift_end_point.point.pose;

      // distance between shoulder lane's left boundary and current lane center
      double distance_road_to_left_boundary = util::getDistanceToShoulderBoundary(
        route_handler.getShoulderLanelets(), road_lane_reference_path.points.back().point.pose);
      // distance between shoulder lane's left boundary and current lane center
      double distance_road_to_target =
        distance_road_to_left_boundary + common_parameter.vehicle_width / 2 + margin;

      shift_point.length = distance_road_to_target;
      path_shifter.addShiftPoint(shift_point);
    }

    // offset front side from reference path
    bool offset_back = false;
    if (!path_shifter.generate(&shifted_path, offset_back)) {
      continue;
    }

    const auto shift_end_idx =
      motion_utils::findNearestIndex(shifted_path.path.points, shift_end_point.point.pose);

    const auto goal_idx = motion_utils::findNearestIndex(shifted_path.path.points, goal_pose);

    if (shift_end_idx && goal_idx) {
      for (size_t i = 0; i < shifted_path.path.points.size(); ++i) {
        auto & point = shifted_path.path.points.at(i);
        if (i < *shift_end_idx) {
          // set velocity during shift
          point.point.longitudinal_velocity_mps = std::min(
            point.point.longitudinal_velocity_mps,
            road_lane_reference_path.points.back().point.longitudinal_velocity_mps);
          continue;
        } else if (i > *goal_idx) {
          // set velocity after goal
          point.point.longitudinal_velocity_mps = 0.0;
          continue;
        }
        point.point.longitudinal_velocity_mps = pull_over_velocity;

        // add closest shoulder lane id
        lanelet::Lanelet closest_shoulder_lanelet;
        lanelet::utils::query::getClosestLanelet(
          route_handler.getShoulderLanelets(), point.point.pose, &closest_shoulder_lanelet);
        point.lane_ids.clear();
        point.lane_ids.push_back(closest_shoulder_lanelet.id());
      }

      candidate_path.straight_path = road_lane_reference_path;
      // resample is needed for adding orientation to path points for collision check
      candidate_path.path = util::resamplePathWithSpline(
        combineReferencePath(road_lane_reference_path, shifted_path.path), 1.0);

      // add goal pose because resampling removes it
      PathPointWithLaneId goal_path_point{};
      goal_path_point.point.pose = goal_pose;
      // z of goal_pose can not be used
      // https://github.com/autowarefoundation/autoware.universe/issues/711
      goal_path_point.point.pose.position.z =
        candidate_path.path.points.back().point.pose.position.z;
      goal_path_point.point.longitudinal_velocity_mps = 0.0;
      goal_path_point.lane_ids = shifted_path.path.points.back().lane_ids;
      candidate_path.path.points.push_back(goal_path_point);

      const auto shift_start_idx =
        motion_utils::findNearestIndex(candidate_path.path.points, shift_point.start.position);
      for (size_t i = shift_start_idx; i < candidate_path.path.points.size(); i++) {
        candidate_path.shifted_path.path.points.push_back(candidate_path.path.points.at(i));
      }

      shift_point.start_idx = motion_utils::findNearestIndex(
        candidate_path.shifted_path.path.points, shift_point.start.position);
      shift_point.end_idx = motion_utils::findNearestIndex(
        candidate_path.shifted_path.path.points, shift_point.end.position);
      // todo: shift_length size dose not match path size due to resample,
      // so sume fuctions (like getTurnInfo()) can not be used with this shifted_point.
      candidate_path.shifted_path.shift_length = shifted_path.shift_length;
      candidate_path.shift_point = shift_point;
      // candidate_path.acceleration = acceleration;
      candidate_path.preparation_length = straight_distance;
      candidate_path.pull_over_length = pull_over_distance;
    } else {
      RCLCPP_ERROR_STREAM(
        rclcpp::get_logger("behavior_path_planner").get_child("pull_over").get_child("util"),
        "lane change end idx not found on target path.");
      continue;
    }

    candidate_paths.push_back(candidate_path);
  }

  return candidate_paths;
}

std::vector<ShiftParkingPath> selectValidPaths(
  const std::vector<ShiftParkingPath> & paths, const lanelet::ConstLanelets & current_lanes,
  const lanelet::ConstLanelets & target_lanes,
  const lanelet::routing::RoutingGraphContainer & overall_graphs, const Pose & current_pose,
  const bool isInGoalRouteSection, const Pose & goal_pose)
{
  std::vector<ShiftParkingPath> available_paths;

  for (const auto & path : paths) {
    if (hasEnoughDistance(
          path, current_lanes, target_lanes, current_pose, isInGoalRouteSection, goal_pose,
          overall_graphs)) {
      available_paths.push_back(path);
    }
  }

  return available_paths;
}

bool selectSafePath(
  const std::vector<ShiftParkingPath> & paths,
  const OccupancyGridBasedCollisionDetector & occupancy_grid_map, ShiftParkingPath & selected_path)
{
  for (const auto & path : paths) {
    if (!occupancy_grid_map.hasObstacleOnPath(path.shifted_path.path, false)) {
      selected_path = path;
      return true;
    }
  }

  // set first path for force pull over if no valid path found
  if (!paths.empty()) {
    selected_path = paths.front();
    return false;
  }

  return false;
}

bool hasEnoughDistance(
  const ShiftParkingPath & path, const lanelet::ConstLanelets & current_lanes,
  [[maybe_unused]] const lanelet::ConstLanelets & target_lanes, const Pose & current_pose,
  const bool isInGoalRouteSection, const Pose & goal_pose,
  [[maybe_unused]] const lanelet::routing::RoutingGraphContainer & overall_graphs)
{
  const double pull_over_prepare_distance = path.preparation_length;
  const double pull_over_distance = path.pull_over_length;
  const double pull_over_total_distance = pull_over_prepare_distance + pull_over_distance;

  if (pull_over_total_distance > util::getDistanceToEndOfLane(current_pose, current_lanes)) {
    return false;
  }

  // if (pull_over_total_distance >
  // util::getDistanceToNextIntersection(current_pose, current_lanes)) {
  //   return false;
  // }

  if (
    isInGoalRouteSection &&
    pull_over_total_distance > util::getSignedDistance(current_pose, goal_pose, current_lanes)) {
    return false;
  }

  // if (
  //   pullover_total_distance >
  //   util::getDistanceToCrosswalk(current_pose, current_lanes, overall_graphs)) {
  //   return false;
  // }
  return true;
}

// Use occupancy grid to check safety instead of this function.
bool isPullOverPathSafe(
  const PathWithLaneId & path, const lanelet::ConstLanelets & current_lanes,
  const lanelet::ConstLanelets & target_lanes,
  const PredictedObjects::ConstSharedPtr dynamic_objects, const Pose & current_pose,
  const Twist & current_twist, const double vehicle_width,
  const PullOverParameters & ros_parameters, const bool use_buffer, const double acceleration)
{
  if (path.points.empty()) {
    return false;
  }
  if (target_lanes.empty() || current_lanes.empty()) {
    return false;
  }
  if (dynamic_objects == nullptr) {
    return true;
  }
  const auto arc = lanelet::utils::getArcCoordinates(current_lanes, current_pose);
  constexpr double check_distance = 100.0;

  // parameters
  const double time_resolution = ros_parameters.prediction_time_resolution;
  const double min_thresh = ros_parameters.min_stop_distance;
  const double stop_time = ros_parameters.stop_time;

  double buffer;
  double lateral_buffer;
  if (use_buffer) {
    buffer = ros_parameters.hysteresis_buffer_distance;
    lateral_buffer = 0.5;
  } else {
    buffer = 0.0;
    lateral_buffer = 0.0;
  }
  double current_lane_check_start_time = 0.0;
  const double current_lane_check_end_time =
    ros_parameters.pull_over_prepare_duration + ros_parameters.pull_over_duration;
  double target_lane_check_start_time = 0.0;
  const double target_lane_check_end_time =
    ros_parameters.pull_over_prepare_duration + ros_parameters.pull_over_duration;
  if (!ros_parameters.enable_collision_check_at_prepare_phase) {
    current_lane_check_start_time = ros_parameters.pull_over_prepare_duration;
    target_lane_check_start_time = ros_parameters.pull_over_prepare_duration;
  }

  // find obstacle in pull_over target lanes
  // retrieve lanes that are merging target lanes as well
  const auto target_lane_object_indices =
    util::filterObjectsByLanelets(*dynamic_objects, target_lanes);

  // find objects in current lane
  const auto current_lane_object_indices_lanelet = util::filterObjectsByLanelets(
    *dynamic_objects, current_lanes, arc.length, arc.length + check_distance);
  const auto current_lane_object_indices = util::filterObjectsByPath(
    *dynamic_objects, current_lane_object_indices_lanelet, path,
    vehicle_width / 2 + lateral_buffer);

  const auto & vehicle_predicted_path = util::convertToPredictedPath(
    path, current_twist, current_pose, target_lane_check_end_time, time_resolution, acceleration);

  // Collision check for objects in current lane
  for (const auto & i : current_lane_object_indices) {
    const auto & obj = dynamic_objects->objects.at(i);
    std::vector<PredictedPath> predicted_paths;
    if (ros_parameters.use_all_predicted_path) {
      predicted_paths.resize(obj.kinematics.predicted_paths.size());
      std::copy(
        obj.kinematics.predicted_paths.begin(), obj.kinematics.predicted_paths.end(),
        predicted_paths.begin());
    } else {
      auto & max_confidence_path = *(std::max_element(
        obj.kinematics.predicted_paths.begin(), obj.kinematics.predicted_paths.end(),
        [](const auto & path1, const auto & path2) {
          return path1.confidence > path2.confidence;
        }));
      predicted_paths.push_back(max_confidence_path);
    }
    for (const auto & obj_path : predicted_paths) {
      double distance = util::getDistanceBetweenPredictedPaths(
        obj_path, vehicle_predicted_path, current_lane_check_start_time,
        current_lane_check_end_time, time_resolution);
      double thresh;
      if (isObjectFront(current_pose, obj.kinematics.initial_pose_with_covariance.pose)) {
        thresh = util::l2Norm(current_twist.linear) * stop_time;
      } else {
        thresh =
          util::l2Norm(obj.kinematics.initial_twist_with_covariance.twist.linear) * stop_time;
      }
      thresh = std::max(thresh, min_thresh);
      thresh += buffer;
      if (distance < thresh) {
        return false;
      }
    }
  }

  // Collision check for objects in pull over target lane
  for (const auto & i : target_lane_object_indices) {
    const auto & obj = dynamic_objects->objects.at(i);
    std::vector<PredictedPath> predicted_paths;
    if (ros_parameters.use_all_predicted_path) {
      predicted_paths.resize(obj.kinematics.predicted_paths.size());
      std::copy(
        obj.kinematics.predicted_paths.begin(), obj.kinematics.predicted_paths.end(),
        predicted_paths.begin());
    } else {
      auto & max_confidence_path = *(std::max_element(
        obj.kinematics.predicted_paths.begin(), obj.kinematics.predicted_paths.end(),
        [](const auto & path1, const auto & path2) {
          return path1.confidence > path2.confidence;
        }));
      predicted_paths.push_back(max_confidence_path);
    }

    bool is_object_in_target = false;
    if (ros_parameters.use_predicted_path_outside_lanelet) {
      is_object_in_target = true;
    } else {
      for (const auto & llt : target_lanes) {
        if (lanelet::utils::isInLanelet(obj.kinematics.initial_pose_with_covariance.pose, llt)) {
          is_object_in_target = true;
        }
      }
    }

    if (is_object_in_target) {
      for (const auto & obj_path : predicted_paths) {
        const double distance = util::getDistanceBetweenPredictedPaths(
          obj_path, vehicle_predicted_path, target_lane_check_start_time,
          target_lane_check_end_time, time_resolution);
        double thresh;
        if (isObjectFront(current_pose, obj.kinematics.initial_pose_with_covariance.pose)) {
          thresh = util::l2Norm(current_twist.linear) * stop_time;
        } else {
          thresh =
            util::l2Norm(obj.kinematics.initial_twist_with_covariance.twist.linear) * stop_time;
        }
        thresh = std::max(thresh, min_thresh);
        thresh += buffer;
        if (distance < thresh) {
          return false;
        }
      }
    } else {
      const double distance = util::getDistanceBetweenPredictedPathAndObject(
        obj, vehicle_predicted_path, target_lane_check_start_time, target_lane_check_end_time,
        time_resolution);
      double thresh = min_thresh;
      if (isObjectFront(current_pose, obj.kinematics.initial_pose_with_covariance.pose)) {
        thresh = std::max(thresh, util::l2Norm(current_twist.linear) * stop_time);
      }
      thresh += buffer;
      if (distance < thresh) {
        return false;
      }
    }
  }

  return true;
}

bool isObjectFront(const Pose & ego_pose, const Pose & obj_pose)
{
  tf2::Transform tf_map2ego, tf_map2obj;
  Pose obj_from_ego;
  tf2::fromMsg(ego_pose, tf_map2ego);
  tf2::fromMsg(obj_pose, tf_map2obj);
  tf2::toMsg(tf_map2ego.inverse() * tf_map2obj, obj_from_ego);

  return obj_from_ego.position.x > 0;
}

}  // namespace pull_over_utils
}  // namespace behavior_path_planner
