#ifndef ROSEFUSION_H
#define ROSEFUSION_H

#include "data_types.h"
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;

namespace rosefusion {

    class Pipeline {
    public:

        Pipeline(const CameraParameters _camera_config,
                 const DataConfiguration _data_config,
                 const ControllerConfiguration _controller_config);

        ~Pipeline() = default;


        bool process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map, cv::Mat& shaded_img);

        void get_poses() const;

        PointCloud extract_pointcloud() const;


    private:
        const CameraParameters camera_parameters;
        const DataConfiguration data_config;
        const ControllerConfiguration controller_config;
        const std::vector<int> particle_leve;

        float iter_tsdf;
        internal::VolumeData volume;
        internal::QuaternionData PST; 
        internal::SearchData search_data;
        internal::FrameData frame_data;
        Eigen::Matrix4d current_pose;
        std::vector<Eigen::Matrix4d> poses;
        bool previous_frame_success=false;
        Matf61da initialize_search_size;
        size_t frame_id;
        cv::Mat last_model_frame;
    };

    void export_ply(const std::string& filename, const PointCloud& point_cloud);



    namespace internal {


        void surface_measurement(const cv::Mat_<cv::Vec3b>& color_map,
                                      const cv::Mat_<float>& depth_map,
                                      FrameData& frame_data,
                                      const CameraParameters& camera_params,
                                      const float depth_cutoff);




        bool pose_estimation(const VolumeData& volume,
                             const QuaternionData& quaternions,
                              SearchData& search_data,
                             Eigen::Matrix4d& pose,
                             FrameData& frame_data,
                             const CameraParameters& cam_params,
                             const ControllerConfiguration& controller_config,
                             const std::vector<int> particle_level,
                             float * iter_tsdf,
                             bool * previous_frame_success,
                             Matf61da& initialize_search_size
                            );
        namespace cuda {


            void surface_reconstruction(const cv::cuda::GpuMat& depth_image,
                                        const cv::cuda::GpuMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params,
                                        const float truncation_distance,
                                        const Eigen::Matrix4d& model_view);


            void surface_prediction(const VolumeData& volume,
                                    cv::cuda::GpuMat& shading_buffer,
                                    const CameraParameters& cam_parameters,
                                    const float truncation_distance,
                                    const float3 init_pos,
                                    cv::Mat& shaded_img,
                                    const Eigen::Matrix4d& pose);

            PointCloud extract_points(const VolumeData& volume, const int buffer_size);

        }

    }
}
#endif 
