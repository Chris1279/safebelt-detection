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

	//++++++++++++++++++++++++++++++++++++�ṹ��++++++++++++++++++++++++++++++++++++

	struct FaceFeature
	{
		int PointX[34];
		int PointY[34];
	};

	//++++++++++++++++++++++++++++++++++++һ�����++++++++++++++++++++++++++++++++++++

	//����ת�ַ���
	string CommenFunction_ReturnString(int intType);

	//�ַ���ת����
	int CommenFunction_ReturnInt(string stringType);

	//�ļ���������JPG�ļ�
	std::vector<string> CommenFunction_GetAllInTheDirectory(string dirpath);

	//�ļ���������JPG�ļ�
	std::vector<string> CommenFunction_GetAllInTheDirectory(string srcPath, SEARCH_FILE_TYPE type);

	//�����ļ���
	bool CommenFunction_CreateDirectory(const std::string folder);

	//���ص�ǰʱ��
	string CommenFunction_ReturnCurrentTime();

	//�ȴ������ڵ���
	void CommenFunction_WhileWaitkey();

	int  CommenFunction_GetMaxIndexOfvectorForDouble(std::vector<double> vectors);

	int  CommenFunction_GetMinIndexOfvectorForDouble(std::vector<double> vectors);

	int  CommenFunction_GetMaxIndexOfvectorForInt(std::vector<int> vectors);

	int  CommenFunction_GetMinIndexOfvectorForInt(std::vector<int> vectors);

	//34��Landmark���ļ���ȡ
	FaceFeature CommenFunction_LoadBinFile(string filePath);

	//34��Landmark���ļ�����
	void CommenFunction_SaveBinFile(string filePath, FaceFeature feature);

	//�����������
	double CommenFunction_DistanceOfTwoPoints(Point p1, Point p2);

	//++++++++++++++++++++++++++++++++++++��ͼ����++++++++++++++++++++++++++++++++++++

	//��ȡ�ļ����µ�����ͼ�񣬲�ת��ΪgrayImg
	std::vector<Mat> BasicImgProc_GetAllImgsInDirectoryAndTransferToGrayImgs(string imgPath, SEARCH_FILE_TYPE type);

	//���㼯������ͼ����
	void BasicImgProc_DisplayPointsInImg(std::vector<std::vector<Point>> Points, Mat originalImg);

	//��ʾͼ��
	void BasicImgProc_ShowImg(Mat img);

	//����ҶȺ�
	double BasicImgProc_CalculateImageGraySum(Mat img);

	//ͼ�������
	Mat BasicImgProc_ImangAndOperation(Mat img1, Mat img2);

	//BGRת��Ϊ�Ҷ�ͼ
	Mat BasicImgProc_ColorToGrayImg(Mat rgbImg);

	//��ͨ��ֵ��
	Mat BasicImgProc_NormalThreshlod(Mat srcImg, int threshold);

	//����Ӧ��ֵ��
	Mat BasicImgProc_AdaptiveThreshold(Mat srcImg, int threshold);

	//����canny������ȡ��Ե
	Mat BasicImgProc_CannyEdge(Mat srcGray, int bigThreshold, int smallThreshold, int operatorSize);

	//����
	void BasicImgProc_VidesToJpegs(const string srcPath, const string dstPath);

	//����ֱ��ͼ����ʾ
	Mat BasicImgProc_CalcHistAndDisplay(Mat grayImg);

	//ROI����
	Mat BasicImgProc_ROICopy(Mat bigMat, Mat smallMat, Rect locationOfBigMat);

	//���Ҳ���ʾ������������������ͼ��
	Mat BasicImgProc_FindAllTheContoursImg(Mat srcImg);

	//��������ͼ������ֱ�ߣ����ػ���ֱ�ߵ�ͼ��
	Mat BasicImgProc_GetAllTheLinesImgP(Mat srcImg, int minLength, Rect mouthRect);

	//��������ͼ������ֱ�ߣ����ػ���ֱ�ߵ�ͼ��
	Mat BasicImgProc_GetAllTheLinesImgStd(Mat srcImg, int minLength);

	//����Sobel�ݶ�ͼ
	Mat BasicImgProc_GetGradientImg(Mat srcImg);

	//�õ�����������
	std::vector<Point> BasicImgProc_GetFacePoints(FaceFeature feature);

	//�õ������������
	cv::Rect BasicImgProc_GetSmokingAreaImg(std::vector<Point> facePoints);

	//�õ��첿����
	cv::Rect BasicImgProc_GetMouthRect(std::vector<Point> facePoints);

	//��ʾ����ͼ���ϣ����Է���
	void BasicImgProc_DisplayPoints(std::vector<Point> points);

	//�ݶ�ͼ���ֵ��
	Mat BasicImgProc_GetGradientBinaryImg(Mat srcImg);

	//ʹ��LSD���߶�
	std::vector<Vec4f> BasicImgProc_GetParallelLinesUsingLSD(Mat imgBinary, int minLength, int maxLength);

	//��ͨ����
	int BasicImgProc_ConnectedAreaImgMark(const cv::Mat& _binImg, cv::Mat& _lableImg);

	//������ͨ����ͼ�񣬵õ���ֵͼ�����е���ͨ��ͼ��
	std::map<int, Mat> BasicImgProc_GetAllConnectedAreaImgs(Mat binaryImg);

	//ϸ��ͼ��
	Mat BasicImg_ProcGetThinImg(cv::Mat& src, int intera);

	//��ȡ��ֵͼ���еİ׵�
	std::vector<Point> BasicImg_GetWhitePointInBinaryImg(Mat binaryImg);

	//��ʱ����ת
	Mat BasicImg_ContrarotateImg(Mat srcImg, double angle);

	//��ӽ�������(��ͨ��ͼ��)
	Mat BasicImg_AddSpiceAndAaltNoise(Mat img, double SNR);

	//��Ӹ�˹��������ͨ��ͼ��
	Mat BasicImg_AddGassianNoise(Mat img, double mu, double sigma, int k, int PixcelMax, int PixcelMin);

	//++++++++++++++++++++++++++++++++++++������ȡ++++++++++++++++++++++++++++++++++++

	//����Hog������ʹ��֮ǰ�ȳ߶ȹ�һ����64*128��
	std::vector<float> FeatureCalculate_GetHog(Mat src);

	//����ͼ���Hog�������ӻ�ͼ��(scaleFactor=3, viz_factor=2.5)
	Mat FeatureCalculate_GetHogVisualImage(Mat src, int scaleFactor, double viz_factor);

	//����Ҷ�ͼ�����������㣬�õ������Ҷ�����ͼ
	Mat BasicImgProc_GetGrayContourFace(Mat imgGray, std::vector<Point> facePoints);

	//����Ҷ�ͼ�����������㣬�õ���������Ҷ�����ͼ
	Mat BasicImgProc_GetGrayContourSmokingAreaOfFace(Mat imgGray, std::vector<Point> facePoints);

	//����Ҷ�ͼ�����������㣬�õ���������
	Mat BasicImgProc_GetGrayContourFaceMask(Mat imgGray, std::vector<Point> facePoints);

	//++++++++++++++++++++++++++++++++++++����ѧ����++++++++++++++++++++++++++++++++++++

	//������˹�����
	double generateGaussianNoise(double mu, double sigma);
}

#pragma once
#pragma once
