// Copyright 2021 TIER IV, Inc.
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

#ifndef CENTERPOINT_CONFIG_HPP_
#define CENTERPOINT_CONFIG_HPP_

#include <cstddef>
#include <vector>

namespace centerpoint
{
class CenterPointConfig
{
public:
  explicit CenterPointConfig(
    const float point_feature_size, const std::size_t max_num_voxels,
    const std::vector<double> & point_cloud_range, const std::vector<double> & voxel_size,
    const std::size_t downsample_factor, const std::size_t encoder_in_feature_size)
  {
    point_feature_size_ = point_feature_size;
    max_num_voxels_ = max_num_voxels;
    if (point_cloud_range.size() == 6) {
      range_min_x_ = static_cast<float>(point_cloud_range[0]);
      range_min_y_ = static_cast<float>(point_cloud_range[1]);
      range_min_z_ = static_cast<float>(point_cloud_range[2]);
      range_max_x_ = static_cast<float>(point_cloud_range[3]);
      range_max_y_ = static_cast<float>(point_cloud_range[4]);
      range_max_z_ = static_cast<float>(point_cloud_range[5]);
    }
    if (voxel_size.size() == 3) {
      voxel_size_x_ = static_cast<float>(voxel_size[0]);
      voxel_size_y_ = static_cast<float>(voxel_size[1]);
      voxel_size_z_ = static_cast<float>(voxel_size[2]);
    }
    downsample_factor_ = downsample_factor;
    encoder_in_feature_size_ = encoder_in_feature_size;

    grid_size_x_ = static_cast<std::size_t>((range_max_x_ - range_min_x_) / voxel_size_x_);
    grid_size_y_ = static_cast<std::size_t>((range_max_y_ - range_min_y_) / voxel_size_y_);
    grid_size_z_ = static_cast<std::size_t>((range_max_z_ - range_min_z_) / voxel_size_z_);
    offset_x_ = range_min_x_ + voxel_size_x_ / 2;
    offset_y_ = range_min_y_ + voxel_size_y_ / 2;
    offset_z_ = range_min_z_ + voxel_size_z_ / 2;
    down_grid_size_x_ = grid_size_x_ / downsample_factor_;
    down_grid_size_y_ = grid_size_y_ / downsample_factor_;
  };

  // input params
  const std::size_t point_dim_size_{3};  // x, y and z
  std::size_t point_feature_size_{4};    // x, y, z and timelag
  std::size_t max_num_points_per_voxel_{32};
  std::size_t max_num_voxels_{40000};
  float range_min_x_{-89.6f};
  float range_min_y_{-89.6f};
  float range_min_z_{-3.0f};
  float range_max_x_{89.6f};
  float range_max_y_{89.6f};
  float range_max_z_{5.0f};
  float voxel_size_x_{0.32f};
  float voxel_size_y_{0.32f};
  float voxel_size_z_{8.0f};

  // network params
  const std::size_t batch_size_{1};
  std::size_t downsample_factor_{2};
  std::size_t encoder_in_feature_size_{9};
  const std::size_t encoder_out_feature_size_{32};
  const std::size_t head_out_size_{6};
  const std::size_t head_out_offset_size_{2};
  const std::size_t head_out_z_size_{1};
  const std::size_t head_out_dim_size_{3};
  const std::size_t head_out_rot_size_{2};
  const std::size_t head_out_vel_size_{2};

  // calculated params
  std::size_t grid_size_x_ = (range_max_x_ - range_min_x_) / voxel_size_x_;
  std::size_t grid_size_y_ = (range_max_y_ - range_min_y_) / voxel_size_y_;
  std::size_t grid_size_z_ = (range_max_z_ - range_min_z_) / voxel_size_z_;
  float offset_x_ = range_min_x_ + voxel_size_x_ / 2;
  float offset_y_ = range_min_y_ + voxel_size_y_ / 2;
  float offset_z_ = range_min_z_ + voxel_size_z_ / 2;
  std::size_t down_grid_size_x_ = grid_size_x_ / downsample_factor_;
  std::size_t down_grid_size_y_ = grid_size_y_ / downsample_factor_;
};

}  // namespace centerpoint

#endif  // CENTERPOINT_CONFIG_HPP_
