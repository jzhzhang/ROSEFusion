#include <rosefusion.h>
#include <iostream>
#include <fstream>

using cv::cuda::GpuMat;
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;

namespace rosefusion {

    Pipeline::Pipeline(const CameraParameters _camera_config,
                       const DataConfiguration _data_config,
                       const ControllerConfiguration _controller_config) :
            camera_parameters(_camera_config), data_config(_data_config),
            controller_config(_controller_config),
            volume(data_config.volume_size, data_config.voxel_scale),
            frame_data(_camera_config.image_height,_camera_config.image_width),
            particle_leve{10240,3072,1024},PST(particle_leve,_controller_config.PST_path),search_data(particle_leve),
            current_pose{}, poses{}, frame_id{0}, last_model_frame{},iter_tsdf{_controller_config.init_fitness}
    {
        current_pose.setIdentity();
        current_pose(0, 3) = data_config.init_pos.x;
        current_pose(1, 3) = data_config.init_pos.y;
        current_pose(2, 3) = data_config.init_pos.z;
    }

    bool Pipeline::process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map, cv::Mat& shaded_img)
    {


        internal::surface_measurement(color_map, depth_map, frame_data, camera_parameters, data_config.depth_cutoff_distance);                                              
        bool tracking_success { true };
        if (frame_id > 0) { 
            tracking_success = internal::pose_estimation(volume,PST,search_data,current_pose, frame_data, 
            camera_parameters,controller_config,particle_leve,&iter_tsdf,&previous_frame_success,initialize_search_size);
        
        }
        if (!tracking_success )
            return false;

        poses.push_back(current_pose);
        internal::cuda::surface_reconstruction(frame_data.depth_map, frame_data.color_map,
                                               volume, camera_parameters, data_config.truncation_distance,
                                               current_pose.inverse());
        if (controller_config.render_surface){
            internal::cuda::surface_prediction(volume,
                                            frame_data.shading_buffer,
                                            camera_parameters, data_config.truncation_distance,
                                            data_config.init_pos,
                                            shaded_img,
                                            current_pose);
        }

        ++frame_id;
        return true;
    }



    void Pipeline::get_poses() const
    {
        Eigen::Matrix4d init_pose=poses[0];
        std::ofstream trajectory;
        trajectory.open(data_config.result_path+data_config.seq_name+".txt");
        std::cout<<data_config.result_path+data_config.seq_name+".txt"<<std::endl;
        int iter_count=0;
        for (auto pose : poses){
            Eigen::Matrix4d temp_pose=init_pose.inverse()*pose;
            Eigen::Matrix3d rotation_m=temp_pose.block(0,0,3,3);
            Eigen::Vector3d translation=temp_pose.block(0,3,3,1)/1000;
            Eigen::Quaterniond q(rotation_m);
            trajectory<<iter_count<<" "<<translation.x()<<" "<<translation.y()<<" "<<translation.z()<<\
            " "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<std::endl;
            iter_count++;
        }
        trajectory.close();
    }

    PointCloud Pipeline::extract_pointcloud() const
    {
        PointCloud cloud_data = internal::cuda::extract_points(volume, data_config.pointcloud_buffer_size);
        return cloud_data;
    }




    void export_ply(const std::string& filename, const PointCloud& point_cloud)
    {
        std::ofstream file_out { filename };
        if (!file_out.is_open())
            return;

        file_out << "ply" << std::endl;
        file_out << "format ascii 1.0" << std::endl;
        file_out << "element vertex " << point_cloud.num_points << std::endl;
        file_out << "property float x" << std::endl;
        file_out << "property float y" << std::endl;
        file_out << "property float z" << std::endl;
        file_out << "property float nx" << std::endl;
        file_out << "property float ny" << std::endl;
        file_out << "property float nz" << std::endl;
        file_out << "property uchar red" << std::endl;
        file_out << "property uchar green" << std::endl;
        file_out << "property uchar blue" << std::endl;
        file_out << "end_header" << std::endl;

        for (int i = 0; i < point_cloud.num_points; ++i) {
            float3 vertex = point_cloud.vertices.ptr<float3>(0)[i];
            float3 normal = point_cloud.normals.ptr<float3>(0)[i];
            uchar3 color = point_cloud.color.ptr<uchar3>(0)[i];
            file_out << vertex.x << " " << vertex.y << " " << vertex.z << " " << normal.x << " " << normal.y << " "
                     << normal.z << " ";
            file_out << static_cast<int>(color.x) << " " << static_cast<int>(color.y) << " "
                     << static_cast<int>(color.z) << std::endl;
        }
    }


}