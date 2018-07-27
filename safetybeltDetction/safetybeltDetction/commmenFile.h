#pragma once
#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

#include <io.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

namespace commenFunction
{
	enum SEARCH_FILE_TYPE
	{
		FILE_JPG = 0,
		FILE_MP4 = 1,
		FILE_BMP = 2,
		FILE_PNG = 3
	};

	//++++++++++++++++++++++++++++++++++++结构体++++++++++++++++++++++++++++++++++++

	struct FaceFeature
	{
		int PointX[34];
		int PointY[34];
	};

	//++++++++++++++++++++++++++++++++++++一般操作++++++++++++++++++++++++++++++++++++

	//整型转字符型
	string CommenFunction_ReturnString(int intType);

	//字符型转整型
	int CommenFunction_ReturnInt(string stringType);

	//文件夹下所有JPG文件
	std::vector<string> CommenFunction_GetAllInTheDirectory(string dirpath);

	//文件夹下所有JPG文件
	std::vector<string> CommenFunction_GetAllInTheDirectory(string srcPath, SEARCH_FILE_TYPE type);

	//创建文件夹
	bool CommenFunction_CreateDirectory(const std::string folder);

	//返回当前时间
	string CommenFunction_ReturnCurrentTime();

	//等待，用于调试
	void CommenFunction_WhileWaitkey();

	int  CommenFunction_GetMaxIndexOfvectorForDouble(std::vector<double> vectors);

	int  CommenFunction_GetMinIndexOfvectorForDouble(std::vector<double> vectors);

	int  CommenFunction_GetMaxIndexOfvectorForInt(std::vector<int> vectors);

	int  CommenFunction_GetMinIndexOfvectorForInt(std::vector<int> vectors);

	//34个Landmark点文件提取
	FaceFeature CommenFunction_LoadBinFile(string filePath);

	//34个Landmark点文件保存
	void CommenFunction_SaveBinFile(string filePath, FaceFeature feature);

	//计算两点距离
	double CommenFunction_DistanceOfTwoPoints(Point p1, Point p2);

	//++++++++++++++++++++++++++++++++++++简单图像处理++++++++++++++++++++++++++++++++++++

	//获取文件夹下的所有图像，并转化为grayImg
	std::vector<Mat> BasicImgProc_GetAllImgsInDirectoryAndTransferToGrayImgs(string imgPath, SEARCH_FILE_TYPE type);

	//将点集绘制在图像上
	void BasicImgProc_DisplayPointsInImg(std::vector<std::vector<Point>> Points, Mat originalImg);

	//显示图像
	void BasicImgProc_ShowImg(Mat img);

	//计算灰度和
	double BasicImgProc_CalculateImageGraySum(Mat img);

	//图像与操作
	Mat BasicImgProc_ImangAndOperation(Mat img1, Mat img2);

	//BGR转化为灰度图
	Mat BasicImgProc_ColorToGrayImg(Mat rgbImg);

	//普通阈值化
	Mat BasicImgProc_NormalThreshlod(Mat srcImg, int threshold);

	//自适应二值化
	Mat BasicImgProc_AdaptiveThreshold(Mat srcImg, int threshold);

	//利用canny算子提取边缘
	Mat BasicImgProc_CannyEdge(Mat srcGray, int bigThreshold, int smallThreshold, int operatorSize);

	//采样
	void BasicImgProc_VidesToJpegs(const string srcPath, const string dstPath);

	//计算直方图并显示
	Mat BasicImgProc_CalcHistAndDisplay(Mat grayImg);

	//ROI拷贝
	Mat BasicImgProc_ROICopy(Mat bigMat, Mat smallMat, Rect locationOfBigMat);

	//查找并显示所有轮廓，返回轮廓图像
	Mat BasicImgProc_FindAllTheContoursImg(Mat srcImg);

	//输入轮廓图，查找直线，返回绘制直线的图像
	Mat BasicImgProc_GetAllTheLinesImgP(Mat srcImg, int minLength, Rect mouthRect);

	//输入轮廓图，查找直线，返回绘制直线的图像
	Mat BasicImgProc_GetAllTheLinesImgStd(Mat srcImg, int minLength);

	//计算Sobel梯度图
	Mat BasicImgProc_GetGradientImg(Mat srcImg);

	//得到脸部轮廓点
	std::vector<Point> BasicImgProc_GetFacePoints(FaceFeature feature);

	//得到抽烟区域矩形
	cv::Rect BasicImgProc_GetSmokingAreaImg(std::vector<Point> facePoints);

	//得到嘴部矩形
	cv::Rect BasicImgProc_GetMouthRect(std::vector<Point> facePoints);

	//显示点在图像上，调试方便
	void BasicImgProc_DisplayPoints(std::vector<Point> points);

	//梯度图像二值化
	Mat BasicImgProc_GetGradientBinaryImg(Mat srcImg);

	//使用LSD找线段
	std::vector<Vec4f> BasicImgProc_GetParallelLinesUsingLSD(Mat imgBinary, int minLength, int maxLength);

	//连通域标记
	int BasicImgProc_ConnectedAreaImgMark(const cv::Mat& _binImg, cv::Mat& _lableImg);

	//根据连通域标记图像，得到二值图像所有的连通域图像
	std::map<int, Mat> BasicImgProc_GetAllConnectedAreaImgs(Mat binaryImg);

	//细化图像
	Mat BasicImg_ProcGetThinImg(cv::Mat& src, int intera);

	//获取二值图像中的白点
	std::vector<Point> BasicImg_GetWhitePointInBinaryImg(Mat binaryImg);

	//逆时针旋转
	Mat BasicImg_ContrarotateImg(Mat srcImg, double angle);

	//添加椒盐噪声(单通道图像)
	Mat BasicImg_AddSpiceAndAaltNoise(Mat img, double SNR);

	//添加高斯噪声（单通道图像）
	Mat BasicImg_AddGassianNoise(Mat img, double mu, double sigma, int k, int PixcelMax, int PixcelMin);

	//++++++++++++++++++++++++++++++++++++特征提取++++++++++++++++++++++++++++++++++++

	//计算Hog特征，使用之前先尺度归一化（64*128）
	std::vector<float> FeatureCalculate_GetHog(Mat src);

	//绘制图像的Hog特征可视化图像(scaleFactor=3, viz_factor=2.5)
	Mat FeatureCalculate_GetHogVisualImage(Mat src, int scaleFactor, double viz_factor);

	//输入灰度图和脸部轮廓点，得到人脸灰度轮廓图
	Mat BasicImgProc_GetGrayContourFace(Mat imgGray, std::vector<Point> facePoints);

	//输入灰度图和脸部轮廓点，得到抽烟区域灰度轮廓图
	Mat BasicImgProc_GetGrayContourSmokingAreaOfFace(Mat imgGray, std::vector<Point> facePoints);

	//输入灰度图和脸部轮廓点，得到人脸掩码
	Mat BasicImgProc_GetGrayContourFaceMask(Mat imgGray, std::vector<Point> facePoints);

	//++++++++++++++++++++++++++++++++++++简单数学计算++++++++++++++++++++++++++++++++++++

	//产生高斯随机数
	double generateGaussianNoise(double mu, double sigma);
}

#pragma once
#pragma once
