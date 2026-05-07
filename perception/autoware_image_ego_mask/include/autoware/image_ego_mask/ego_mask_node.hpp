#ifndef AUTOWARE__IMAGE_EGO_MASK__EGO_MASK_NODE_HPP_
#define AUTOWARE__IMAGE_EGO_MASK__EGO_MASK_NODE_HPP_

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace autoware::image_preprocessor
{

class EgoMaskNode final : public rclcpp::Node
{
public:
  explicit EgoMaskNode(const rclcpp::NodeOptions & options);

private:
  struct PolygonSpec
  {
    // points: [x0, y0, x1, y1, ...]
    std::vector<double> points;
    bool normalized{false};
  };

  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  cv::Scalar fillScalarForEncoding(const std::string & encoding) const;
  void applyMask(cv::Mat & image) const;

  image_transport::Subscriber sub_;
  image_transport::Publisher pub_;

  std::string input_transport_;
  std::string output_transport_;

  std::vector<PolygonSpec> polygons_;
  std::vector<double> fill_value_bgr_;
  bool enabled_{true};
};

}  // namespace autoware::image_preprocessor

#endif  // AUTOWARE__IMAGE_EGO_MASK__EGO_MASK_NODE_HPP_

