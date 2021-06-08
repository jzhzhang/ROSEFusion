#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char* argv[]){

    string seq_name;
    string association_file;
    string output_path;
    string source_path;
    int img_height;
    int img_width;
    double depth_ratio;
    bool color_to_depth;


    std::cout<<"Read configure file\n";
    const std::string config_file(argv[1]);
    printf("%s\n",config_file.c_str());
    cv::FileStorage configSetting(config_file.c_str(),cv::FileStorage::READ);
    configSetting["seq_name"]>>seq_name;
    configSetting["association_file"]>>association_file;
    configSetting["output_path"]>>output_path;
    configSetting["source_path"]>>source_path;
    img_height=configSetting["img_height"];
    img_width=configSetting["img_width"];
    depth_ratio=configSetting["depth_ratio"];
    color_to_depth=bool(int(configSetting["color_to_depth"]));
    


    FILE * fp;
    fp = fopen((output_path+seq_name+".seq").c_str(),"wb");




 
    ifstream asso_file(association_file);
    string line;
    string color_file;
    string depth_file;
    string color_name;
    string depth_name;
    int FrameNum=0;

    vector<cv::Mat> v_depth;
    vector<cv::Mat> v_color;

    while(getline(asso_file,line)){
        istringstream iss(line);
        printf("%d\n",FrameNum);

        if (color_to_depth){
            iss>>color_name>>color_file>>depth_name>>depth_file;  
        }else{
            iss>>depth_name>>depth_file>>color_name>>color_file;  
        }

        cv::Mat color_img=cv::imread((source_path+color_file).c_str());
        cv::Mat depth_map=cv::imread((source_path+depth_file).c_str(),-1);

        depth_map.convertTo(depth_map,CV_16UC1,1000*1.0/depth_ratio);
        v_depth.push_back(depth_map);
        v_color.push_back(color_img);

        FrameNum++;
    }

    fwrite(&FrameNum,sizeof(int32_t),1,fp);
    fwrite(&img_height,sizeof(int32_t),1,fp);
    fwrite(&img_width,sizeof(int32_t),1,fp);

    for (int i=0;i<FrameNum;i++){

        fwrite((char*) v_depth[i].data,img_height*img_width*2,1,fp);
        fwrite((uchar*)v_color[i].data,img_height*img_width*3,1,fp);
    }
    fclose(fp);

}
