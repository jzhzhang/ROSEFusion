#include <rosefusion.h>

using Matf31da = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;
using Matrix3frm = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;


namespace rosefusion {
    namespace internal {

        namespace cuda { 
            void estimate_step(const Eigen::Matrix3f& rotation_current, const Matf31da& translation_current,
                               const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
                               const Eigen::Matrix3f& rotation_previous_inv, const Matf31da& translation_previous,
                               const CameraParameters& cam_params,
                               const cv::cuda::GpuMat& vertex_map_previous, const cv::cuda::GpuMat& normal_map_previous,
                               float distance_threshold, float angle_threshold,
                               Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b);

            bool particle_evaluation(const VolumeData& volume,const QuaternionData& quaterinons, SearchData& search_data ,
                                const Eigen::Matrix3d& rotation_current, const Matf31da& translation_current,
                                const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
                                const Eigen::Matrix3d& rotation_previous_inv, const Matf31da& translation_previous,
                                const CameraParameters& cam_params, const int particle_index,const int particle_size,
                                const Matf61da& search_size,const int resolution_level,const int level_index,
                                Eigen::Matrix<double, 7, 1>& mean_transform, float * tsdf
                                );
        }

 
        void update_seach_size(const float tsdf, const float scaling_coefficient,Matf61da& search_size, Eigen::Matrix<double, 7, 1>& mean_transform )
        {
            
            double s_tx=fabs(mean_transform(0,0))+1e-3;
            double s_ty=fabs(mean_transform(1,0))+1e-3;
            double s_tz=fabs(mean_transform(2,0))+1e-3; 

            double s_qx=fabs(mean_transform(4,0))+1e-3; 
            double s_qy=fabs(mean_transform(5,0))+1e-3;
            double s_qz=fabs(mean_transform(6,0))+1e-3;

            double trans_norm=sqrt(s_tx*s_tx+s_ty*s_ty+s_tz*s_tz+s_qx*s_qx+s_qy*s_qy+s_qz*s_qz);


            double normal_tx=s_tx/trans_norm;
            double normal_ty=s_ty/trans_norm;
            double normal_tz=s_tz/trans_norm;
            double normal_qx=s_qx/trans_norm;
            double normal_qy=s_qy/trans_norm;
            double normal_qz=s_qz/trans_norm;

            search_size(3,0) = scaling_coefficient * tsdf*normal_qx+1e-3;
            search_size(4,0) = scaling_coefficient * tsdf*normal_qy+1e-3;
            search_size(5,0) = scaling_coefficient * tsdf*normal_qz+1e-3;  
            search_size(0,0) = scaling_coefficient * tsdf*normal_tx+1e-3;
            search_size(1,0) = scaling_coefficient * tsdf*normal_ty+1e-3;
            search_size(2,0) = scaling_coefficient * tsdf*normal_tz+1e-3;
        }


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
                             Matf61da& initialize_search_size)
        {   

            Eigen::Matrix3d current_global_rotation = pose.block(0, 0, 3, 3);
            Eigen::Vector3d current_global_translation = pose.block(0, 3, 3, 1);
            Eigen::Matrix3d previous_global_rotation_inverse(current_global_rotation.inverse());
            Eigen::Vector3d previous_global_translation = pose.block(0, 3, 3, 1);


            float beta=controller_config.momentum;
            Matf61da previous_search_size;


            Matf61da search_size;
            if (*previous_frame_success && controller_config.scaling_inherit_directly){
                search_size<< initialize_search_size(0,0),
                initialize_search_size(1,0),
                initialize_search_size(3,0),
                initialize_search_size(4,0),
                initialize_search_size(5,0),
                initialize_search_size(0,0);

            }else{
                float lens= controller_config.scaling_coefficient1*(*iter_tsdf);
                search_size<< lens, lens, lens, lens, lens, lens;

            }


            int particle_index[20] ={0,1+20,2+40,3,4+20,5+40,6+0,7+20,8+40,
                                    9+0,10+20,11+40,12+0,13+20,14+40,
                                    15+0,16+20,17+40,18+0,19+20};
            int level[20] = {32,16,8,32,16,8,32,16,8,32,16,8,32,16,8,32,16,8,32,16};


            int count_particle=0;
            int level_index=5;
            bool success=true;
            bool previous_success=true;

            int count=0;
            int count_success=0;
            float min_tsdf;

            double qx;
            double qy;
            double qz;

            while(true){

                Eigen::Matrix<double, 7, 1> mean_transform=Eigen::Matrix<double, 7, 1>::Zero();
  
                if(count==controller_config.max_iteration){
                    break;
                }

                if (!success){
                    count_particle=0;
                }
                success=cuda::particle_evaluation(volume,quaternions,search_data,current_global_rotation, current_global_translation,
                                    frame_data.vertex_map, frame_data.normal_map,
                                    previous_global_rotation_inverse, previous_global_translation,
                                    cam_params,particle_index[count_particle],particle_level[particle_index[count_particle]/20],
                                    search_size,level[count_particle],level_index,
                                    mean_transform,&min_tsdf); 


                if (count==0 && !success)
                {
                    *iter_tsdf=min_tsdf;
                }
            
                qx=mean_transform(4,0);
                qy=mean_transform(5,0);
                qz=mean_transform(6,0);

                if (success){
                    if (count_particle<19){
                        ++count_particle;

                    }
                    ++count_success;

                    auto camera_translation_incremental = mean_transform.head<3>();
                    double qw=mean_transform(3,0);
                    Eigen::Matrix3d camera_rotation_incremental;

                    camera_rotation_incremental << 1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw),
                                                    2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw),
                                                    2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy);

                    current_global_translation =  current_global_translation + camera_translation_incremental*1000;
                    current_global_rotation = camera_rotation_incremental * current_global_rotation;
          
                }
                

                level_index+=5;
                level_index=level_index%level[count_particle];

                update_seach_size(min_tsdf,controller_config.scaling_coefficient2,search_size,mean_transform);

                if (previous_success&&success){
                    search_size(0,0)=beta*search_size(0,0)+(1-beta)*previous_search_size(0,0);
                    search_size(1,0)=beta*search_size(1,0)+(1-beta)*previous_search_size(1,0);
                    search_size(2,0)=beta*search_size(2,0)+(1-beta)*previous_search_size(2,0);
                    search_size(3,0)=beta*search_size(3,0)+(1-beta)*previous_search_size(3,0);
                    search_size(4,0)=beta*search_size(4,0)+(1-beta)*previous_search_size(4,0);
                    search_size(5,0)=beta*search_size(5,0)+(1-beta)*previous_search_size(5,0);  

                    previous_search_size<<search_size(0,0),search_size(1,0),search_size(2,0),
                    search_size(3,0),search_size(4,0),search_size(5,0);

                }else if(success){

                    previous_search_size<<search_size(0,0),search_size(1,0),search_size(2,0),
                    search_size(3,0),search_size(4,0),search_size(5,0);

                }

                if(success){
                    previous_success=true;
                }else{
                    previous_success=false;
                }

                if(count==0){
                    if (success){
                        initialize_search_size<<search_size(0,0),search_size(1,0),search_size(2,0),
                        search_size(3,0),search_size(4,0),search_size(5,0);        
                        *previous_frame_success=true;
                    }else{
                        *previous_frame_success=false;
                    }
                }
                ++count;
            }

            if (count_success==0 ){
                return false;
            }
            pose.block(0, 0, 3, 3) = current_global_rotation;
            pose.block(0, 3, 3, 1) = current_global_translation;

            return true;
        }
    }
}
