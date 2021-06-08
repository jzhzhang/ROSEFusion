#ifndef DATAREADER_H
#define DATAREADER_H


#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>



class DataReader
{
    public:
    DataReader(std::string file, bool flipColors);
    virtual ~DataReader();
    int getFramesNum();
    void getNextFrame(cv::Mat& Color_mat,cv::Mat& Depth_mat);
    bool hasMore();

    private:
    std::vector<cv::Mat> v_color;
    std::vector<cv::Mat> v_depth;

    void readFile();
    FILE * fp;
    int numFrames;
    int height;
    int width;
    int currentFrame;

};

#endif