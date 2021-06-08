#include <rosefusion.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#pragma GCC diagnostic pop

using cv::cuda::GpuMat;

namespace rosefusion {
    namespace internal {

        namespace cuda { 
            void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, const float depth_cutoff,
                                    const CameraParameters cam_params);
            void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map);
        }

        void surface_measurement(const cv::Mat_<cv::Vec3b>& color_frame,
                                      const cv::Mat_<float>& depth_frame,
                                      FrameData& frame_data,
                                      const CameraParameters& camera_params,
                                      const float depth_cutoff)
        {

            frame_data.color_map.upload(color_frame);
            frame_data.depth_map.upload(depth_frame);
            cuda::compute_vertex_map(frame_data.depth_map, frame_data.vertex_map,
                                     depth_cutoff, camera_params);
            cuda::compute_normal_map(frame_data.vertex_map, frame_data.normal_map);

        }
    }
}