cmake_minimum_required(VERSION 3.0)

project(rosefusion)

set(CMAKE_CXX_STANDARD 14)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------

find_package(Pangolin REQUIRED)



## CUDA
find_package(CUDA 11.0 REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    #  Specify the arch of your CUDA hardware here
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++14 -gencode arch=compute_75,code=sm_75)
ENDIF ()

## OpenCV
find_package(OpenCV 4.5.0  REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    message(${OpenCV_INCLUDE_DIRS})
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

## Eigen3
find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
endif (EIGEN3_INCLUDE_DIR)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")


set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)


add_executable(ROSEFusion ${PROJECT_SOURCE_DIR}/main.cpp )
add_executable(seq_gen ${PROJECT_SOURCE_DIR}/seq_gen.cpp )



include_directories("${PROJECT_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/cuda/include")
file(GLOB ROSEFusion_SRCS "${PROJECT_SOURCE_DIR}/cuda/*.cu" "${PROJECT_SOURCE_DIR}/*.cpp")

cuda_add_library(ROSEFusion_lib STATIC "${ROSEFusion_SRCS}")

target_link_libraries(ROSEFusion "${OpenCV_LIBS}" ROSEFusion_lib)
target_link_libraries(ROSEFusion ${Pangolin_LIBRARIES})

target_link_libraries(seq_gen "${OpenCV_LIBS}")
