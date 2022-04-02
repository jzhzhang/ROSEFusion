# ROSEFusion :rose:

This project is based on our SIGGRAPH 2021 paper, [ROSEFusion: **R**andom **O**ptimization for Online Den**SE** Reconstruction under Fast Camera Motion
](https://arxiv.org/abs/2105.05600).



## Introduction

ROSEFusion is proposed to tackle the difficulties in fast-motion camera tracking using random optimization with depth information only. Our method performs robust  camera tracking under fast camera motion at a real-time frame rate, without loop closure or global pose optimization.

 <p id="demo1" align="center"> 
  <img src="assets/intro.gif" />
 </p>

## Installation
Our code is based on C++ and CUDA with the support of:
- [Pangolin](https://github.com/stevenlovegrove/Pangolin) (tested on v0.6)
- OpenCV with CUDA (v.4.5 is required, for instance you can follow the [link](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7))  
- Eigen (tested on 3.3.9)
- CUDA (tested with V11.1, 11.4)

Please make sure the architecture ```(sm_xx and compute_xx)``` in the [L22 of CMakeLists.txt](CMakeLists.txt#L22) is compatible with your own graphics card.


Our code has been tested with Nvidia GeForce RTX 2080 SUPER on Ubuntu 16.04. 

## [Option] Test with Docker

We have already upload a docker image with all the lib, code and data. You can download the docker image from the [google drive](https://drive.google.com/file/d/1sNvm8vSJM5MxxDpgkqDo-FhpwXBdraMb/view?usp=sharing).

### **Prepare**
Please make sure you have successfully installed the [docker](https://www.docker.com/) and [nvidia docker](github.com/NVIDIA/nvidia-docker). and once the environment is ready, you can use following commands to boot the docker image:
```
sudo docker load -i rosefusion_docker.tar 
sudo docker run -it  --gpus all jiazhao/rosefusion:v7 /bin/bash
```


And please check the architecture in the L22 of ``` /home/code/ROSEFusion-main/CMakeList.txt``` is compatible with your own graphics card. If not, change the sm_xx and compute_xx, then rebuild the ROSEFusion.


### **QuickStart**
We have already configured the path and data in the docker image. You can simply run "run_example.sh" and "run_stairwell.sh" at  ```/home/code/ROSEFusion-main/build``` and the trajectory and reconstuciton would be saved in ```/home/code/rosefusion_xxx_data```. 



## Configuration File
We use the following configuration files to make the parameters setting easier. There are four types of configuration files.

- **seq_generation_config.yaml:** data information 
- **camera_config.yaml:** camera and image information.
- **data_config.yaml:** output path, sequence file path and parameters of the volume.
- **controller_config.yaml:** visualization, results saving and parameters of tracking.

The **seq_generation_config.yaml** is only used for data preparation, and the other three types of configuration files are necessary to run the ROSEFusion. We have alreay prepared some configuration files of some common datasets, you can check the details in `[type]_config/` directory. You can change the parameters to fit your own dataset.

## Data Preparation
The details of data preparation can be found in [src/seq_gen.cpp](src/seq_gen.cpp). By using the *seq_generation_config.yaml* introduced above, you can run the script as:
```
./seq_gen  sequence_information.yaml
```
Once finished, there would be a `.seq` file which could be used for future reconstruction.


## Particle Swarm Template
We share the same pre-sampled PST as our paper. Each PST is saved as an N×6 image and the N means the number of particles. You can find the ``.tiff`` images in [PST dicrectory](/PST), and please change the PST path in ``controller_config.yaml `` with your own path.

## Running
Finally, to run the ROSEFusion, you need to provide the `camera_config.yaml`, `data_config.yaml` and `controller_config.yaml`. We already share configuration files of many common datasets in `./camera_config`, `./data_config`, `/controller_config`. All the parameters of configuration files can be modified as you want. Once you have all the required files, you can run the ROSEFsuion as:
```
./ROSEFsuion  your_camera_config.yaml your_data_config.yaml your_controller_config.yaml
```
For a quick start, you can download and use a small size synthesis [seq file with related configuration files](https://drive.google.com/drive/folders/1vW5GV2xsJN1kIrl-5JZX1fUrpjCtp5AS?usp=sharing). Here is a preview.


 <p id="demo1" align="center"> 
  <img src="assets/example.gif" />
 </p>

## FastCaMo Dataset
We present the **Fast** **Ca**mera **Mo**tion dataset, which contains both synthetic and real captured sequences. For more details, please refer to the paper.
### FastCaMo-Synth
With 10 diverse room-scale scenes from [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset), we render the color images and depth maps along the synthetic trajectories. The raw sequences are provided in [FastCaMo-synth-data(raw).zip](https://drive.google.com/file/d/15PG7jd1wFdf26zaPp-pq04RAoCRQA1qt/view?usp=sharing), and we also provide the [FastCaMo-synth-data(noise).zip](https://drive.google.com/file/d/1a8QLimLFvteac6OfGPxSsTqITrJSC2ox/view?usp=sharing) with synthetic noise and motion blur. We use the same noise model as [simkinect](https://github.com/ankurhanda/simkinect). For evaluation, you can download the ground truth [trajectories](https://drive.google.com/file/d/106p9N99K-X3_jbt8PRcKthySbpxhIwxB/view?usp=sharing).

### FastCaMo-Real
It contains 12 [real captured RGB-D sequences](https://drive.google.com/drive/folders/1kDUz_Vxjy5zi5LO8G5HwjkZ0WbetsBy1?usp=sharing) under fast camera motions. Each sequence is recorded in a challenging scene like gym or stairwell by using [Azure Kinect DK](https://azure.microsoft.com/en-us/services/kinect-dk/). We provide accurate dense reconstructions as ground truth, which are modeled with the high-end laser scanner. However, the original models are extremely large, and we utilized the built-in spatial downsample algorithm from cloudcompare. You can download the sub-sampled [models of FastCaMo-real form here](https://drive.google.com/drive/folders/1AXiTpC-UZ0WLhLCqjfKEPl5H40B0NccI?/usp=sharing). 

 <p id="demo1" align="center"> 
  <img src="assets/fastcamo-real.gif" />
 </p>

## Citation
If you find our work useful in your research, please consider citing:
```
@article {zhang_sig21,
    title = {ROSEFusion: Random Optimization for Online Dense Reconstruction under Fast Camera Motion},
    author = {Jiazhao Zhang and Chenyang Zhu and Lintao Zheng and Kai Xu},
    journal = {ACM Transactions on Graphics (SIGGRAPH 2021)},
    volume = {40},
    number = {4},
    year = {2021}
}
```

## Acknowledgments
Our code is inspired by [KinectFusionLib](https://github.com/chrdiller/KinectFusionLib).

This is an open-source version of ROSEFusion, some functions have been rewritten to avoid certain license. It would not be expected to reproduce the result exactly, but the result is almost the same.
## License
The source code is released under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) license.

## Contact
If you have any questions, feel free to email Jiazhao Zhang at zhngjizh@gmail.com.

