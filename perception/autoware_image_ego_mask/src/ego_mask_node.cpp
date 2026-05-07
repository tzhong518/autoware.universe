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

#include <cv_bridge/cv_bridge.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
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

}  // namespace

EgoMaskNode::EgoMaskNode(const rclcpp::NodeOptions & options)
: Node("ego_mask", options)
{
  enabled_ = declare_parameter<bool>("enabled", true);

  input_transport_ = declare_parameter<std::string>("input_transport", "raw");
  output_transport_ = declare_parameter<std::string>("output_transport", "raw");

  fill_value_bgr_ = declare_parameter<std::vector<double>>("fill_value_bgr", {0.0, 0.0, 0.0});
  if (fill_value_bgr_.size() != 3) {
    throw std::runtime_error("Parameter 'fill_value_bgr' must have 3 elements.");
  }

  // polygons: list of flattened point arrays, each as [x0,y0,x1,y1,...]
  // polygons_normalized: optional list<bool> same size; if true interpret points as [0..1] ratios.
  const auto polygons = declare_parameter<std::vector<std::vector<double>>>("polygons", {});
  const auto polygons_normalized = declare_parameter<std::vector<bool>>("polygons_normalized", {});
  if (!polygons_normalized.empty() && polygons_normalized.size() != polygons.size()) {
    throw std::runtime_error("Parameter 'polygons_normalized' must be empty or same size as 'polygons'.");
  }
  polygons_.reserve(polygons.size());
  for (size_t i = 0; i < polygons.size(); ++i) {
    PolygonSpec p;
    p.points = polygons[i];
    if (p.points.size() < 6 || (p.points.size() % 2) != 0) {
      throw std::runtime_error("Each polygon in 'polygons' must have even length >= 6.");
    }
    p.normalized = polygons_normalized.empty() ? false : polygons_normalized[i];
    polygons_.push_back(std::move(p));
  }

  pub_ = image_transport::create_publisher(this, "~/output/image", rmw_qos_profile_sensor_data);
  sub_ = image_transport::create_subscription(
    this, "~/input/image", std::bind(&EgoMaskNode::onImage, this, std::placeholders::_1),
    input_transport_, rmw_qos_profile_sensor_data);
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

