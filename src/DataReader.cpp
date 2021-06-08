#include "DataReader.h"
#include <opencv2/opencv.hpp>



DataReader::DataReader(std::string file,bool flipColors)
{
    fp = fopen(file.c_str(),"rb");
    currentFrame=0;
    auto tmp = fread(&numFrames,sizeof(int32_t),1,fp);
    assert(tmp);
    tmp = fread(&height,sizeof(int32_t),1,fp);
    assert(tmp);
    tmp = fread(&width,sizeof(int32_t),1,fp);
    assert(tmp);
    readFile();
}

DataReader::~DataReader()
{
    fclose(fp);
}

void DataReader::readFile()
{
    for (int i=0;i<numFrames;i++){
        ushort depth_buffer[height*width];
        uchar color_buffer[height*width*3];
        auto tmp = fread(depth_buffer,height*width*2,1,fp);
        assert(tmp);
        tmp = fread(color_buffer,height*width*3,1,fp);
        assert(tmp);

        v_color.push_back(cv::Mat(height,width,CV_8UC3,color_buffer).clone());
        v_depth.push_back(cv::Mat(height,width,CV_16UC1,depth_buffer).clone());

    }

}

void DataReader::getNextFrame(cv::Mat& Color_mat,cv::Mat& Depth_mat)
{   
    Color_mat=v_color[currentFrame];
    Depth_mat=v_depth[currentFrame];
    currentFrame++;
}

bool DataReader::hasMore(){
    return (currentFrame<numFrames);
}

