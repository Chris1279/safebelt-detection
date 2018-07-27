#include <dlib\opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <opencv2\opencv.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

#include <iostream>  
#include <time.h>
#include <map> 
#include <stack>

#include "safebeltDetection.h"
#include "commonFile.h"

using namespace cv;
using namespace dlib;
using namespace std;
using namespace commenFunction;

//算法检测模块
void SafebeltDetectModule(VideoCapture capture)
{
	//初始化
	std::vector<Point> facePoints;
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor shaperModel;
	dlib::deserialize("sp.dat") >> shaperModel;

	//抽烟检测模块
	SafebeltDetect temp;

	int index = 0;
	//更改
	long currentFrame = 1;
	//获取图像
	while (1)
	{
		//输出图像
		Mat srcImg;
		capture >> srcImg;
		Mat finalImg = srcImg.clone();
		//如果为空退出
		if (srcImg.empty())
		{
			break;
		}
		else
		{
			//检测人脸
			Mat imgGray;
			cvtColor(srcImg, imgGray, CV_BGR2GRAY);
			cv_image<unsigned char> cimg(imgGray);
			std::vector<dlib::rectangle> faces = detector(cimg);

			//检测到人脸
			if (faces.size() > 0)
			{
				//---------------------------得到landmark点集
				std::vector<full_object_detection> shapes;

				for (unsigned long i = 0; i < faces.size(); ++i)
					shapes.push_back(shaperModel(cimg, faces[i]));

				for (unsigned long i = 0; i <= 33; ++i)
				{
					int x = shapes[0].part(i).x();
					int y = shapes[0].part(i).y();
					facePoints.push_back(Point(x, y));
				}

				//------------------------------------------------------------------------在这里检测是否系安全带	
				bool ifSafebelt = temp.DetectIfSafebelt(srcImg, facePoints);

				//显示二值化图像
				Mat binaryIMg = temp.GetBinaryImg();

				//安全带中间结果
				Mat midSafebeltImg = temp.GetmidResultSafebeltImg();
				Mat midSafebeltImg1 = temp.GetmidResultSafebeltImg1();
				Mat midSafebeltImg2 = temp.GetmidResultSafebeltImg2();

				Mat midSafebeltshowImg, midSafebeltshowImg1, midSafebeltshowImg2;
				resize(midSafebeltImg, midSafebeltshowImg, Size(0, 0), 0.5, 0.5);
				resize(midSafebeltImg1, midSafebeltshowImg1, Size(0, 0), 0.5, 0.5);
				resize(midSafebeltImg2, midSafebeltshowImg2, Size(0, 0), 0.5, 0.5);

				finalImg = temp.GetfinalResultImg();

				imshow("中间结果—所有平行线安全带", midSafebeltshowImg);
				imshow("中间结果—距离删除", midSafebeltshowImg1);
				imshow("中间结果—最大交集", midSafebeltshowImg2);

				//保存每一帧的抽烟区域图像
				//stringstream str, str1, str2;
				//str  << "G:\\dataset\\safebelt\\" << "allParalle" << currentFrame<< ifSafebelt << ".png";
				//imwrite(str.str(), midSafebeltImg);
				//str1 << "G:\\dataset\\safebelt\\" << "distance" << currentFrame << ifSafebelt << ".png";
				//imwrite(str1.str(), midSafebeltImg1);
				//str2 << "G:\\dataset\\safebelt\\" << "inrect" << currentFrame << ifSafebelt << ".png";
				//imwrite(str2.str(), midSafebeltImg2);

				currentFrame++;

				if (ifSafebelt == true)
				{

					putText(finalImg, "Safebelt", Point(20, 20), 1, 2, Scalar(255), 2);
					////保存图像
					string currentTime = CommenFunction_ReturnCurrentTime();
					string imgPath = "G:\\dataset\\safebelt\\result\\" + currentTime + ".bmp";
					cv::imwrite(imgPath, finalImg);

				}
				else
					putText(finalImg, "noSafebelt", Point(20, 20), 1, 2, Scalar(255), 2);
			}
			else
			{
				//putText(imgGray, "Noface", Point(20, 20), 1, 2, Scalar(0), 2);
				putText(finalImg, "Noface", Point(20, 20), 1, 2, Scalar(0), 2);

			}
			//更改放大五倍后缩小十倍
			Mat imgShow;
			resize(finalImg, imgShow, Size(0, 0), 1, 1);
			imshow("1", imgShow);

			//------------------------------------------------------------------------end

			facePoints.clear();
		}
		cv::waitKey(1);
	}
}

//使用当地视频测试算法(整个文件夹)
void TestAllTheVideos(string videoPath)
{
	string endOfFile = "*.mp4";
	std::vector<string> files = CommenFunction_GetAllInTheDirectory(videoPath + endOfFile);
	//std::vector<string> files = CommenFunction_GetAllInTheDirectory(videoPath, FILE_JPG);

	for (int i = 0; i < files.size(); i++)
	{
		cout << i << "\n";
		string filePath = videoPath + files[i];
		VideoCapture capture(filePath);
		SafebeltDetectModule(capture);
	}
}

//主函数
void main()
{
	//测试文件夹下的所有视频
	TestAllTheVideos("G:\\testVideo\\safebelt\\鲁HR9333\\");

	////测试摄像头输入
	//SmokingDetectModule(VideoCapture(0));	

	//TestFaceAlign();
}
