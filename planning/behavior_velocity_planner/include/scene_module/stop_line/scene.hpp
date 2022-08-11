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

#ifndef SCENE_MODULE__STOP_LINE__SCENE_HPP_
#define SCENE_MODULE__STOP_LINE__SCENE_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <lanelet2_extension/utility/query.hpp>
#include <rclcpp/rclcpp.hpp>
#include <scene_module/scene_module_interface.hpp>
#include <utilization/util.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>

namespace behavior_velocity_planner
{

using autoware_auto_planning_msgs::msg::PathWithLaneId;
using tier4_planning_msgs::msg::StopFactor;
using tier4_planning_msgs::msg::StopReason;

class StopLineModule : public SceneModuleInterface
{
public:
  enum class State { APPROACH, STOPPED, START };

  struct DebugData
  {
    double base_link2front;
    boost::optional<geometry_msgs::msg::Pose> stop_pose;
    LineString2d search_stopline;
  };

  struct PlannerParam
  {
    double stop_margin;
    double stop_duration_sec;
    double hold_stop_margin_distance;
    bool use_initialization_stop_line_state;
  };

public:
  StopLineModule(
    const int64_t module_id, const size_t lane_id, const lanelet::ConstLineString3d & stop_line,
    const PlannerParam & planner_param, const rclcpp::Logger logger,
    const rclcpp::Clock::SharedPtr clock);

  bool modifyPathVelocity(PathWithLaneId * path, StopReason * stop_reason) override;

  visualization_msgs::msg::MarkerArray createDebugMarkerArray() override;
  visualization_msgs::msg::MarkerArray createVirtualWallMarkerArray() override;

private:
  int64_t module_id_;

  void insertStopPoint(const geometry_msgs::msg::Point & stop_point, PathWithLaneId & path) const;

  std::shared_ptr<const rclcpp::Time> stopped_time_;

  lanelet::ConstLineString3d stop_line_;

  int64_t lane_id_;

  // State machine
  State state_;

  // Parameter
  PlannerParam planner_param_;

  // Debug
  DebugData debug_data_;
};
}  // namespace behavior_velocity_planner

#endif  // SCENE_MODULE__STOP_LINE__SCENE_HPP_
