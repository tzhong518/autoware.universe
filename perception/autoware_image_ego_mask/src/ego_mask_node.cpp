// Copyright 2026
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

#include "autoware/image_ego_mask/ego_mask_node.hpp"

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>  // for ROS 2 Jazzy or newer
#else
#include <cv_bridge/cv_bridge.h>  // for ROS 2 Humble or older
#endif

#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace autoware::image_preprocessor
{

namespace
{
bool isMono(const std::string & enc)
{
  return enc == sensor_msgs::image_encodings::MONO8 || enc == sensor_msgs::image_encodings::TYPE_8UC1;
}

bool isBgrLike(const std::string & enc)
{
  return enc == sensor_msgs::image_encodings::BGR8 || enc == sensor_msgs::image_encodings::RGB8 ||
         enc == sensor_msgs::image_encodings::TYPE_8UC3;
}

std::string loadTextFile(const std::string & path)
{
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Could not open polygons YAML file: " + path);
  }
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

struct ParsedPolygon
{
  std::vector<double> points;
  bool normalized{false};
};

/** Root YAML mapping:
 *  polygons: list of [x0,y0,...] OR list of {points: [...], normalized: bool}
 *  polygons_normalized: optional bool list (parallel to polygons when using simple lists)
 */
std::vector<ParsedPolygon> parsePolygonsYaml(const std::string & yaml_text)
{
  const auto trim_empty = [](const std::string & s) -> bool {
    return s.find_first_not_of(" \t\n\r\f\v") == std::string::npos;
  };
  if (yaml_text.empty() || trim_empty(yaml_text)) {
    return {};
  }

  YAML::Node root = YAML::Load(yaml_text);
  if (!root.IsMap()) {
    throw std::runtime_error("polygons YAML: root must be a mapping (e.g. 'polygons: [...]').");
  }

  std::vector<bool> normalized_parallel;
  if (root["polygons_normalized"]) {
    const YAML::Node nn = root["polygons_normalized"];
    if (!nn.IsSequence()) {
      throw std::runtime_error("polygons YAML: 'polygons_normalized' must be a sequence of booleans.");
    }
    for (const auto & item : nn) {
      normalized_parallel.push_back(item.as<bool>());
    }
  }

  const YAML::Node polys = root["polygons"];
  if (!polys.IsDefined() || polys.IsNull()) {
    return {};
  }
  if (!polys.IsSequence()) {
    throw std::runtime_error("polygons YAML: 'polygons' must be a sequence.");
  }

  if (!normalized_parallel.empty() && normalized_parallel.size() != polys.size()) {
    throw std::runtime_error(
      "polygons YAML: 'polygons_normalized' must be the same length as 'polygons' when provided.");
  }

  std::vector<ParsedPolygon> out;
  out.reserve(polys.size());
  for (std::size_t i = 0; i < polys.size(); ++i) {
    const YAML::Node & pn = polys[i];
    ParsedPolygon spec{};

    if (pn.IsSequence()) {
      for (const auto & v : pn) {
        spec.points.push_back(v.as<double>());
      }
      if (i < normalized_parallel.size()) {
        spec.normalized = normalized_parallel[i];
      }
    } else if (pn.IsMap()) {
      if (!pn["points"] || !pn["points"].IsSequence()) {
        throw std::runtime_error("polygons YAML: each map entry must have a 'points' sequence.");
      }
      for (const auto & v : pn["points"]) {
        spec.points.push_back(v.as<double>());
      }
      if (pn["normalized"]) {
        spec.normalized = pn["normalized"].as<bool>();
      } else if (i < normalized_parallel.size()) {
        spec.normalized = normalized_parallel[i];
      }
    } else {
      throw std::runtime_error("polygons YAML: each polygon must be a number sequence or a map with 'points'.");
    }

    if (spec.points.size() < 6 || (spec.points.size() % 2) != 0) {
      throw std::runtime_error(
        "polygons YAML: each polygon must have an even length >= 6 (at least 3 (x,y) points).");
    }
    out.push_back(std::move(spec));
  }
  return out;
}

}  // namespace

EgoMaskNode::EgoMaskNode(const rclcpp::NodeOptions & options)
: Node("ego_mask", options)
{
  enabled_ = declare_parameter<bool>("enabled", true);

  input_transport_ = declare_parameter<std::string>("input_transport", "raw");
  output_transport_ = declare_parameter<std::string>("output_transport", "raw");

  // Absolute or relative image_transport base topic (no /compressed suffix).
  // If empty, defaults to private ~/input/image and ~/output/image (remaps may not
  // apply to image_transport plugin subscriptions; prefer these params for wiring).
  const std::string input_base_param =
    declare_parameter<std::string>("input_image_base_topic", "");
  const std::string output_base_param =
    declare_parameter<std::string>("output_image_base_topic", "");
  const std::string in_base =
    input_base_param.empty() ? std::string("~/input/image") : input_base_param;
  const std::string out_base =
    output_base_param.empty() ? std::string("~/output/image") : output_base_param;

  fill_value_bgr_ = declare_parameter<std::vector<double>>("fill_value_bgr", {0.0, 0.0, 0.0});
  if (fill_value_bgr_.size() != 3) {
    throw std::runtime_error("Parameter 'fill_value_bgr' must have 3 elements.");
  }

  // Polygon geometry is YAML-only: use polygons_yaml_file (path) and/or polygons_yaml (inline body).
  // If polygons_yaml_file is non-empty it is read first; otherwise polygons_yaml is parsed.
  const std::string yaml_file = declare_parameter<std::string>("polygons_yaml_file", "");
  const std::string yaml_inline = declare_parameter<std::string>("polygons_yaml", "");
  std::string yaml_body;
  if (!yaml_file.empty()) {
    yaml_body = loadTextFile(yaml_file);
  } else if (!yaml_inline.empty()) {
    yaml_body = yaml_inline;
  }
  const auto parsed = parsePolygonsYaml(yaml_body);
  polygons_.reserve(parsed.size());
  for (const auto & item : parsed) {
    PolygonSpec p;
    p.points = item.points;
    p.normalized = item.normalized;
    polygons_.push_back(std::move(p));
  }

  pub_ = image_transport::create_publisher(this, out_base, rmw_qos_profile_sensor_data);
  sub_ = image_transport::create_subscription(
    this, in_base, std::bind(&EgoMaskNode::onImage, this, std::placeholders::_1), input_transport_,
    rmw_qos_profile_sensor_data);
  RCLCPP_INFO(
    get_logger(), "image_transport subscribe base '%s' (%s), publish base '%s' (%s)", in_base.c_str(),
    input_transport_.c_str(), out_base.c_str(), output_transport_.c_str());
}

cv::Scalar EgoMaskNode::fillScalarForEncoding(const std::string & encoding) const
{
  if (isMono(encoding)) {
    const double gray = std::clamp(fill_value_bgr_.at(0), 0.0, 255.0);
    return cv::Scalar(gray);
  }
  if (isBgrLike(encoding)) {
    const double b = std::clamp(fill_value_bgr_.at(0), 0.0, 255.0);
    const double g = std::clamp(fill_value_bgr_.at(1), 0.0, 255.0);
    const double r = std::clamp(fill_value_bgr_.at(2), 0.0, 255.0);
    return cv::Scalar(b, g, r);
  }
  // Fallback: treat as 8UC3-like; if not, caller may have converted.
  const double b = std::clamp(fill_value_bgr_.at(0), 0.0, 255.0);
  const double g = std::clamp(fill_value_bgr_.at(1), 0.0, 255.0);
  const double r = std::clamp(fill_value_bgr_.at(2), 0.0, 255.0);
  return cv::Scalar(b, g, r);
}

void EgoMaskNode::applyMask(cv::Mat & image) const
{
  if (polygons_.empty()) return;

  std::vector<std::vector<cv::Point>> polys;
  polys.reserve(polygons_.size());

  const int w = image.cols;
  const int h = image.rows;

  for (const auto & poly : polygons_) {
    std::vector<cv::Point> pts;
    pts.reserve(poly.points.size() / 2);
    for (size_t i = 0; i < poly.points.size(); i += 2) {
      double x = poly.points[i];
      double y = poly.points[i + 1];
      if (poly.normalized) {
        x *= static_cast<double>(w);
        y *= static_cast<double>(h);
      }
      pts.emplace_back(static_cast<int>(x), static_cast<int>(y));
    }
    polys.push_back(std::move(pts));
  }

  const cv::Scalar fill = fillScalarForEncoding(
    image.channels() == 1 ? sensor_msgs::image_encodings::MONO8 : sensor_msgs::image_encodings::BGR8);
  cv::fillPoly(image, polys, fill, cv::LINE_AA);
}

void EgoMaskNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  if (!enabled_ || polygons_.empty()) {
    pub_.publish(msg);
    return;
  }

  cv_bridge::CvImageConstPtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvShare(msg, msg->encoding);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_WARN(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat working;
  std::string out_encoding = msg->encoding;

  if (isMono(msg->encoding) || isBgrLike(msg->encoding)) {
    working = in_image_ptr->image.clone();
  } else {
    try {
      auto converted = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      working = converted->image;
      out_encoding = sensor_msgs::image_encodings::BGR8;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN(get_logger(), "Unsupported encoding '%s': %s", msg->encoding.c_str(), e.what());
      return;
    }
  }

  applyMask(working);

  auto out = cv_bridge::CvImage(msg->header, out_encoding, working).toImageMsg();
  pub_.publish(out);
}

}  // namespace autoware::image_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::image_preprocessor::EgoMaskNode)

