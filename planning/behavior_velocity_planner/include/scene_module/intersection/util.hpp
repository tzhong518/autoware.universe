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

#ifndef SCENE_MODULE__INTERSECTION__UTIL_HPP_
#define SCENE_MODULE__INTERSECTION__UTIL_HPP_

#include <rclcpp/rclcpp.hpp>
#include <scene_module/intersection/scene_intersection.hpp>

#include <geometry_msgs/msg/point.hpp>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <memory>
#include <string>
#include <vector>

namespace behavior_velocity_planner
{
namespace util
{
bool setVelocityFrom(
  const size_t idx, const double vel, autoware_auto_planning_msgs::msg::PathWithLaneId * input);

int insertPoint(
  const geometry_msgs::msg::Pose & in_pose,
  autoware_auto_planning_msgs::msg::PathWithLaneId * inout_path);

geometry_msgs::msg::Pose getAheadPose(
  const size_t start_idx, const double ahead_dist,
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path);

bool isAheadOf(const geometry_msgs::msg::Pose & target, const geometry_msgs::msg::Pose & origin);
bool hasLaneId(const autoware_auto_planning_msgs::msg::PathPointWithLaneId & p, const int id);
bool hasDuplicatedPoint(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path,
  const geometry_msgs::msg::Point & point, int * duplicated_point_idx);

/**
 * @brief get objective polygons for detection area
 */
bool getObjectiveLanelets(
  lanelet::LaneletMapConstPtr lanelet_map_ptr, lanelet::routing::RoutingGraphPtr routing_graph_ptr,
  const int lane_id, const double detection_area_length, const double right_margin,
  const double left_margin, std::vector<lanelet::ConstLanelets> * conflicting_lanelets_result,
  lanelet::ConstLanelets * objective_lanelets_result,
  std::vector<lanelet::ConstLanelets> * objective_lanelets_with_margin_result,
  const rclcpp::Logger logger);

/**
 * @brief Generate a stop line and insert it into the path. If the stop line is defined in the map,
 * read it from the map; otherwise, generate a stop line at a position where it will not collide.
 * @param detection_areas used to generate stop line
 * @param original_path   ego-car lane
 * @param target_path     target lane to insert stop point (part of ego-car lane or same to ego-car
 * lane)
 * @param stop_line_idx   generated stop line index
 * @param pass_judge_line_idx  generated stop line index
 * @return false when generation failed
 */
bool generateStopLine(
  const int lane_id, const std::vector<lanelet::CompoundPolygon3d> detection_areas,
  const std::shared_ptr<const PlannerData> & planner_data, const double stop_line_margin,
  autoware_auto_planning_msgs::msg::PathWithLaneId * original_path,
  const autoware_auto_planning_msgs::msg::PathWithLaneId & target_path, int * stop_line_idx,
  int * pass_judge_line_idx, int * first_idx_inside_lane, const rclcpp::Logger logger);

/**
 * @brief If use_stuck_stopline is true, a stop line is generated before the intersection.
 * @param input_path      input path
 * @param output_path     output path
 * @param stuck_stop_line_idx   generated stuck stop line index
 * @param pass_judge_line_idx  generated pass judge line index
 * @return false when generation failed
 */
bool generateStopLineBeforeIntersection(
  const int lane_id, lanelet::LaneletMapConstPtr lanelet_map_ptr,
  const std::shared_ptr<const PlannerData> & planner_data,
  const autoware_auto_planning_msgs::msg::PathWithLaneId & input_path,
  autoware_auto_planning_msgs::msg::PathWithLaneId * output_path, int * stuck_stop_line_idx,
  int * pass_judge_line_idx, const rclcpp::Logger logger);

/**
 * @brief Calculate first path index that is in the polygon.
 * @param path     target path
 * @param polygons target polygon
 * @return path point index
 */
int getFirstPointInsidePolygons(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path,
  const std::vector<lanelet::CompoundPolygon3d> & polygons);

/**
 * @brief Get stop point from map if exists
 * @param stop_pose stop point defined on map
 * @return true when the stop point is defined on map.
 */
bool getStopPoseIndexFromMap(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path, const int lane_id,
  const std::shared_ptr<const PlannerData> & planner_data, int & stop_idx_ip, int dist_thr,
  const rclcpp::Logger logger);

std::vector<lanelet::CompoundPolygon3d> getPolygon3dFromLaneletsVec(
  const std::vector<lanelet::ConstLanelets> & ll_vec, double clip_length);

std::vector<lanelet::CompoundPolygon3d> getPolygon3dFromLanelets(
  const lanelet::ConstLanelets & ll_vec, double clip_length);

std::vector<int> getLaneletIdsFromLaneletsVec(const std::vector<lanelet::ConstLanelets> & ll_vec);

double calcArcLengthFromPath(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & input_path, const size_t src_idx,
  const size_t dst_idx);

lanelet::ConstLanelet generateOffsetLanelet(
  const lanelet::ConstLanelet lanelet, double right_margin, double left_margin);

geometry_msgs::msg::Pose toPose(const geometry_msgs::msg::Point & p);

}  // namespace util
}  // namespace behavior_velocity_planner

#endif  // SCENE_MODULE__INTERSECTION__UTIL_HPP_
