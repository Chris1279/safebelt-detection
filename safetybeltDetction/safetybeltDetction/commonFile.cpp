#include <opencv2\opencv.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

#include <io.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>    
#include <cstdlib>    
#include <Windows.h>   
#include <fcntl.h>
#include <vector>
#include <stdio.h>
#include <direct.h>  
#include <cmath>
#include "commonFile.h"
#include <stack>



using namespace cv;
using namespace std;

namespace commenFunction
{
#pragma region CommenOperation

	//����ʱ��
	void TestTimeSpan()
	{
		double timeStart = (double)getTickCount();

		double nTime = (((double)getTickCount() - timeStart) / getTickFrequency()) / 1000;
		cout << "����������򹲺�ʱ��" << nTime << "����\n" << endl;
	}

	//����ת�ַ���
	string CommenFunction_ReturnString(int intType)
	{
		stringstream sout;
		string s;
		sout << intType;
		s = sout.str();
		return s;
	}

	//�ַ���ת����
	int CommenFunction_ReturnInt(string stringType)
	{
		return atoi(stringType.c_str());
	}

	//�ļ���������JPG�ļ�
	std::vector<string> CommenFunction_GetAllInTheDirectory(string srcPath)
	{
		string thisSrcPath = srcPath;
		const char *		dirpath = thisSrcPath.c_str();

		std::vector<string> fileList;
		intptr_t			handle;
		_finddata_t			findData;


		handle = _findfirst(dirpath, &findData);    // ����Ŀ¼�еĵ�һ���ļ�

		if (handle == -1)
		{
			cout << "Failed to find first file!\n";
		}
		fileList.push_back(findData.name);

		while (1)
		{
			if (_findnext(handle, &findData) == 0)
			{
				fileList.push_back(findData.name);
			}
			else
			{
				break;
			}
		}
		return fileList;
	}

	//�ļ���������JPG�ļ�
	std::vector<string> CommenFunction_GetAllInTheDirectory(string srcPath, SEARCH_FILE_TYPE type)
	{
		string thisSrcPath = "";
		std::vector<string> fileList;

		//string jpgString = "jpg";
		//string bmpString = "bmp";
		//string pngString = "png";
		//string::size_type idx;


		//if (srcPath.find(jpgString) != string::npos)//�����ڡ�
		//{
		//	thisSrcPath = srcPath + "*.jpg";
		//}
		//else if (srcPath.find(bmpString) != string::npos)
		//{
		//	thisSrcPath = srcPath + "*.bmp";
		//}
		//else if (srcPath.find(pngString) != string::npos)
		//{
		//	thisSrcPath = srcPath + "*.png";
		//}
		//else
		//{
		//	return fileList;
		//}



		if (type == FILE_MP4)
		{
			thisSrcPath = srcPath + "*.mp4";
		}
		else if (type == FILE_JPG)
		{
			thisSrcPath = srcPath + "*.jpg";
		}
		else if (type == FILE_BMP)
		{
			thisSrcPath = srcPath + "*.bmp";
		}
		else if (type == FILE_PNG)
		{
			thisSrcPath = srcPath + "*.png";
		}



		const char *		dirpath = thisSrcPath.c_str();


		intptr_t			handle;
		_finddata_t			findData;


		handle = _findfirst(dirpath, &findData);    // ����Ŀ¼�еĵ�һ���ļ�

		if (handle == -1)
		{
			cout << "Failed to find first file!\n";
			return fileList;
		}
		fileList.push_back(findData.name);

		while (1)
		{
			if (_findnext(handle, &findData) == 0)
			{
				fileList.push_back(findData.name);
			}
			else
			{
				break;
			}
		}
		return fileList;
	}

	//���ص�ǰʱ��
	string CommenFunction_ReturnCurrentTime()
	{
		time_t tt = time(NULL);//��䷵�ص�ֻ��һ��ʱ��cuo
		tm* t = localtime(&tt);
		string imgName = commenFunction::CommenFunction_ReturnString(t->tm_mday) + "_" +
			commenFunction::CommenFunction_ReturnString(t->tm_hour) + "_" +
			commenFunction::CommenFunction_ReturnString(t->tm_min) + "_" +
			commenFunction::CommenFunction_ReturnString(t->tm_sec);
		return imgName;
	}

	//�����ļ���
	bool CommenFunction_CreateDirectory(const std::string folder)
	{
		std::string folder_builder;
		std::string sub;
		sub.reserve(folder.size());
		for (auto it = folder.begin(); it != folder.end(); ++it)
		{
			//cout << *(folder.end()-1) << endl;  
			const char c = *it;
			sub.push_back(c);
			if (c == '\\' || it == folder.end() - 1)
			{
				folder_builder.append(sub);
				if (0 != ::_access(folder_builder.c_str(), 0))
				{
					// this folder not exist  
					if (0 != ::_mkdir(folder_builder.c_str()))
					{
						// create failed  
						return false;
					}
				}
				sub.clear();
			}
		}
		return true;
	}

	//����ʹ�ã��ȴ�
	void CommenFunction_WhileWaitkey()
	{
		while (1)
		{
			waitKey(1);
		}
	}



	//����vector<>��������������
	int  CommenFunction_GetMaxIndexOfvectorForInt(std::vector<int> vectors)
	{
		std::vector<int> vectorsNew;
		for (int i = 0; i < vectors.size(); i++)
		{
			vectorsNew.push_back(vectors[i]);
		}

		sort(vectors.begin(), vectors.end());
		int maxIndex = -1;
		for (int i = 0; i < vectors.size(); i++)
		{
			if (vectorsNew[i] == vectors[vectors.size() - 1])
			{
				maxIndex = i;
				break;
			}
		}
		return maxIndex;
	}

	//����vector<>��������С������
	int  CommenFunction_GetMinIndexOfvectorForInt(std::vector<int> vectors)
	{
		std::vector<int> vectorsNew;
		for (int i = 0; i < vectors.size(); i++)
		{
			vectorsNew.push_back(vectors[i]);
		}

		sort(vectors.begin(), vectors.end());
		int minIndex = -1;
		for (int i = 0; i < vectors.size(); i++)
		{
			if (vectorsNew[i] == vectors[0])
			{
				minIndex = i;
				break;
			}
		}
		return minIndex;
	}

	//����vector<>��������������
	int  CommenFunction_GetMaxIndexOfvectorForDouble(std::vector<double> vectors)
	{
		std::vector<double> vectorsNew;
		for (int i = 0; i < vectors.size(); i++)
		{
			vectorsNew.push_back(vectors[i]);
		}

		sort(vectors.begin(), vectors.end());
		int maxIndex = -1;
		for (int i = 0; i < vectors.size(); i++)
		{
			if (vectorsNew[i] == vectors[vectors.size() - 1])
			{
				maxIndex = i;
				break;
			}
		}
		return maxIndex;
	}

	//����vector<>��������С������
	int  CommenFunction_GetMinIndexOfvectorForDouble(std::vector<double> vectors)
	{
		std::vector<double> vectorsNew;
		for (int i = 0; i < vectors.size(); i++)
		{
			vectorsNew.push_back(vectors[i]);
		}

		sort(vectors.begin(), vectors.end());
		int minIndex = -1;
		for (int i = 0; i < vectors.size(); i++)
		{
			if (vectorsNew[i] == vectors[0])
			{
				minIndex = i;
				break;
			}
		}
		return minIndex;
	}

	//34��Landmark���ļ���ȡ
	FaceFeature CommenFunction_LoadBinFile(string filePath)
	{
		FaceFeature temp;
		ifstream is(filePath, ios_base::in | ios_base::binary);
		if (is)
		{
			is.read(reinterpret_cast<char *>(&temp), sizeof(temp));
		}

		is.close();
		return temp;
	}

	//34��landmark���ļ�����
	void CommenFunction_SaveBinFile(string filePath, FaceFeature feature)
	{
		ofstream os(filePath, ios_base::out | ios_base::binary);
		os.write(reinterpret_cast<char *>(&feature), sizeof(feature));
		os.close();
	}

	//�����������
	double CommenFunction_DistanceOfTwoPoints(Point p1, Point p2)
	{
		double sum = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
		return sqrt(sum);
	}

	//������������ֵ
	int CommenFunction_GetMaxValueOfArray(int a[9])
	{
		int temp;
		temp = a[0];
		for (int i = 1; i<9; i++)
		{

			if (a[i]>temp)
				temp = a[i];
		}
		return temp;
	}

	//�����������Сֵ
	int CommenFunction_GetMinValueOfArray(int a[9])
	{
		int temp2;
		temp2 = a[0];
		for (int i = 1; i<9; i++)
		{

			if (a[i]<temp2)
				temp2 = a[i];
		}
		return temp2;
	}

	//����㼯��������һ������ĳ���
	int CommenFunction_GetMaxLength(std::vector<Point> testPoints, std::vector<Point> & dstPoints)
	{
		//�������ھ���
		std::vector<int> testDistance;
		std::vector<Point> xianglinPointsIndexList;
		for (int i = 0; i < testPoints.size() - 1; i++)
		{
			double distance = (testPoints[i + 1].x - testPoints[i].x)*(testPoints[i + 1].x - testPoints[i].x) + (testPoints[i + 1].y - testPoints[i].y)*(testPoints[i + 1].y - testPoints[i].y);

			testDistance.push_back(sqrt(distance));

		}
		std::vector<int> testres;


		for (int i = 0; i < testDistance.size(); i++)
		{

			if (testDistance[i] <= 7)
			{
				testres.push_back(0);
			}
			else
			{
				testres.push_back(1);
			}
		}

		if (sum(testres) != Scalar(0))
		{
			std::vector<int> testNum;
			testNum.push_back(0);
			for (int i = 0; i < testres.size() - 1; i++)
			{

				if (testres[i + 1] - testres[i] == 1)
				{
					testNum.push_back(i);
				}
				if (testres[i + 1] - testres[i] == -1)
				{
					testNum.push_back(i);
				}
			}
			testNum.push_back(testres.size());

			std::vector<int> distNum;

			for (int i = 0; i < testNum.size() / 2; i++)
			{

				Point p(testNum[2 * i + 1], testNum[2 * i]);
				xianglinPointsIndexList.push_back(p);
				distNum.push_back(testNum[2 * i + 1] - testNum[2 * i]);
			}

			int maxDistance = CommenFunction_GetMaxIndexOfvectorForInt(distNum);
			for (int j = xianglinPointsIndexList[maxDistance].y; j < xianglinPointsIndexList[maxDistance].x; j++)
			{
				dstPoints.push_back(testPoints[j]);
			}

			return distNum[maxDistance];
		}
		else
		{
			return testres.size();
		}
	}

	//��������Point֮��ľ���
	double CommenFunction_CalculateSquareDistance(Point p1, Point p2)
	{
		return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
	}

	//�������������С���εĳ���,���س���
	double CommenFunction_GetContourRectLength(vector<Point> contour, std::vector<Point> & points)
	{
		Vec2f temp;

		//������������С����
		RotatedRect box = minAreaRect(contour);
		Point2f vertex[4];
		box.points(vertex);

		temp[0] = CommenFunction_CalculateSquareDistance(vertex[0], vertex[1]);
		temp[1] = CommenFunction_CalculateSquareDistance(vertex[1], vertex[2]);

		if (temp[0] > temp[1])
		{
			points.push_back(vertex[0]);
			points.push_back(vertex[1]);
			return temp[0];
		}
		else
		{
			points.push_back(vertex[1]);
			points.push_back(vertex[2]);
			return temp[1];
		}
	}

	//����vector<>��������������
	int   CommenFunction_GetMaxIndexOfvector(std::vector<double> vectors)
	{
		std::vector<double> vectorsNew;
		for (int i = 0; i < vectors.size(); i++)
		{
			vectorsNew.push_back(vectors[i]);
		}

		sort(vectors.begin(), vectors.end());
		int maxIndex = -1;
		for (int i = 0; i < vectors.size(); i++)
		{
			if (vectorsNew[i] == vectors[vectors.size() - 1])
			{
				maxIndex = i;
				break;
			}
		}
		return maxIndex;
	}

#pragma endregion

#pragma region BasicImgProc

	//��ȡ�ļ����µ�����ͼ�񣬲�ת��ΪgrayImg
	std::vector<Mat> BasicImgProc_GetAllImgsInDirectoryAndTransferToGrayImgs(string imgPath, SEARCH_FILE_TYPE type)
	{
		std::vector<Mat> grayImgList;
		std::vector<string> filesList = CommenFunction_GetAllInTheDirectory(imgPath, type);
		for (int i = 0; i < filesList.size(); i++)
		{
			string imgFile = imgPath + filesList[i];
			Mat srcImg = imread(imgFile);
			Mat grayImg = BasicImgProc_ColorToGrayImg(srcImg);

			grayImgList.push_back(grayImg);
		}

		return grayImgList;
	}

	//���㼯������ͼ����
	void BasicImgProc_DisplayPointsInImg(std::vector<std::vector<Point>> Points, Mat originalImg)
	{
		Mat colorImg;
		if (originalImg.channels() == 3)
		{
			colorImg = originalImg.clone();
		}
		else
		{
			cvtColor(originalImg, colorImg, CV_GRAY2BGR);
		}

		for (int i = 0; i < Points.size(); i++)
		{
			for (int j = 0; j < Points[i].size(); j++)
			{
				circle(colorImg, Points[i][j], 1, Scalar(0, 0, 255), 1);
			}

		}

		imshow("testImg", colorImg);
	}

	//��ʾͼ��
	void BasicImgProc_ShowImg(Mat img)
	{
		string currentTime = CommenFunction_ReturnCurrentTime();
		imshow(currentTime, img);
	}

	//����ͼ�����лҶȺ�
	double BasicImgProc_CalculateImageGraySum(Mat img)
	{
		double sum = 0;
		int rows = img.rows;
		int cols = img.cols;
		for (int i = 0; i<rows; i++)
		{
			for (int j = 0; j<cols; j++)
			{
				uchar t;
				if (img.channels() == 1)
				{
					t = img.at<uchar>(i, j);
					sum = sum + t;
				}
				else if (img.channels() == 3)
				{
					for (int k = 0; k<3; k++)
					{
						t = img.at<Vec3b>(i, j)[k];
						img.at<Vec3b>(i, j)[k] = img.at<Vec3b>(i, cols - 1 - j)[k];
						img.at<Vec3b>(i, cols - 1 - j)[k] = t;
					}
				}
			}
		}

		return sum;
	}

	//ͼ�������
	Mat BasicImgProc_ImangAndOperation(Mat img1, Mat img2)
	{
		Mat dstImg;
		bitwise_and(img1, img2, dstImg);
		return dstImg;
	}

	//BGRת��Ϊ�Ҷ�ͼ
	Mat BasicImgProc_ColorToGrayImg(Mat rgbImg)
	{
		Mat grayImg;
		cvtColor(rgbImg, grayImg, CV_BGR2GRAY);
		return grayImg;
	}

	//��ͨ��ֵ��
	Mat BasicImgProc_NormalThreshlod(Mat srcImg, int threshold)
	{
		Mat binaryImg;
		cv::threshold(srcImg, binaryImg, threshold, 255, THRESH_BINARY);//��ͨȫ����ֵ��
		return binaryImg;
	}

	//����Ӧ��ֵ��
	Mat BasicImgProc_AdaptiveThreshold(Mat srcImg, int threshold)
	{
		Mat binaryImg;
		cv::adaptiveThreshold(srcImg, binaryImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 5);
		return binaryImg;
	}

	//����canny������ȡ��Ե
	Mat BasicImgProc_CannyEdge(Mat srcGray, int bigThreshold, int smallThreshold, int operatorSize)
	{
		Mat cannyImg;
		Canny(srcGray, cannyImg, bigThreshold, smallThreshold, operatorSize);
		return cannyImg;
	}

	//���� 
	void BasicImgProc_VidesToJpegs(const string srcPath, const string dstPath)
	{
		//ͼ����
		string allPath = srcPath + "*.mp4";

		std::vector<string> fileList = CommenFunction_GetAllInTheDirectory(allPath);
		int ImgIndex = 0;
		for (int i = 0; i < fileList.size(); i++)
		{
			//�ļ�·��
			string filePath = srcPath + fileList[i];

			//�����ļ�
			VideoCapture capture(filePath);

			//����Ƶ��
			int index = 0;

			//ѭ���ɼ�
			while (1)
			{
				//��ǰ֡
				Mat frame;
				capture >> frame;

				//���Ϊ���˳�
				if (frame.empty())
				{
					break;
				}
				else
				{
					//����Ƶ��
					index++;
					if (index == 3)
					{
						//ͼƬ����
						ImgIndex++;

						cout << i << "��" << ImgIndex << "\n";

						index = 0;

						//ͼƬ·��
						string jpgPath = dstPath + CommenFunction_ReturnString(ImgIndex) + ".jpg";

						//����
						imwrite(jpgPath, frame);

						//��ʾ
						namedWindow("1");
						resizeWindow("1", 10, 10);
						imshow("1", frame);
					}
				}
			}
		}

		cout << "finished";
	}

	//����ֱ��ͼ����ʾ
	Mat BasicImgProc_CalcHistAndDisplay(Mat grayImg)
	{
		Mat img;

		if (grayImg.channels() == 3)
		{
			cvtColor(grayImg, img, CV_BGR2GRAY);
		}
		else
		{
			img = grayImg.clone();
		}

		/// �趨bin��Ŀ
		int histSize = 255;

		/// �趨ȡֵ��Χ ( R,G,B) )
		float range[] = { 1, 255 };
		const float* histRange = { range };

		bool uniform = true; bool accumulate = false;

		Mat hist;

		/// ����ֱ��ͼ:
		calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		// ����ֱ��ͼ����
		int hist_w = 400;
		int hist_h = 400;
		int bin_w = cvRound((double)hist_w / histSize);

		Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

		/// ��ֱ��ͼ��һ������Χ [ 0, histImage.rows ]
		normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		/// ��ֱ��ͼ�����ϻ���ֱ��ͼ
		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
		}

		//����
		return histImage;
	}

	//ROI����
	Mat BasicImgProc_ROICopy(Mat bigMat, Mat smallMat, Rect locationOfBigMat)
	{
		Mat imageROI = bigMat(locationOfBigMat);

		Mat mask;
		if (smallMat.channels() == 3)
		{
			cvtColor(smallMat, mask, CV_BGR2GRAY);
		}
		else
		{
			mask = smallMat;
		}

		smallMat.copyTo(imageROI, mask);

		return bigMat;
	}

	//���Ҳ���ʾ������������������ͼ��
	Mat BasicImgProc_FindAllTheContoursImg(Mat srcImg)
	{
		//��������
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(srcImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		Mat img(srcImg.size(), CV_8UC1, Scalar(0));
		for (int i = 0; i < contours.size(); i++)
		{
			drawContours(img, contours, i, Scalar(255), 1, 8, hierarchy);
		}

		return img;
	}

	//��������ͼ������ֱ�ߣ����ػ���ֱ�ߵ�ͼ��
	Mat BasicImgProc_GetAllTheLinesImgStd(Mat srcImg, int minLength)
	{
		//hough�߼��
		vector<Vec2f> lines;//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
		HoughLines(srcImg, lines, 1, CV_PI / 180, minLength, 0, 0);

		std::vector<Point> testPoints;
		std::vector<int> lengthList;
		std::vector<std::vector<Point>> filterPoints;

		Mat blank1;
		cvtColor(srcImg.clone(), blank1, CV_GRAY2RGB);

		std::vector<double> distanceList;
		std::vector<vector<Point>> allMaxPoints;
		std::vector<double> list;
		std::vector<Point> longLengthPoints;
		std::vector<vector<Point>> allPoints;

		//��������ֱ�ߣ�����ͼ���е�ÿһ�����ص�	
		for (size_t i = 0; i < lines.size(); i++)
		{
			//��ͼ��
			Mat blank(srcImg.size(), CV_8UC1, Scalar(0));

			//�ҵ���ֱ�ߵĲ���
			float rho = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);

			//��ÿһ��ֱ�ߣ�����ͼ�������еķ���㣬����õ���ֱ���ϣ��򽫸õ��¼����
			int rows = srcImg.rows;
			int cols = srcImg.cols;

			for (int k = 0; k<rows; k++)
			{
				for (int j = 0; j<cols; j++)
				{
					uchar t;
					t = srcImg.at<uchar>(k, j);
					if (t == 255)
					{
						if (abs(rho - j*a - k*b) <= 0.5)
						{
							testPoints.push_back(Point(j, k));
						}
					}

				}
			}

			//���㼯�����ڻ�ͼ����
			for (int j = 0; j < testPoints.size(); j++)
			{
				circle(blank, testPoints[j], 1, Scalar(255), 1);
			}

			//�ڻ�ͼ���в����������������������С���εĳ���
			Mat cl = blank.clone();

			//��������
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			cv::findContours(cl, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

			//����ÿһ�����������������������С���εĳ���
			for (int i = 0; i < contours.size(); i++)
			{
				double longDis = CommenFunction_GetContourRectLength(contours[i], longLengthPoints);
				allPoints.push_back(longLengthPoints);
				distanceList.push_back(longDis);
				longLengthPoints.clear();
			}

			int  maxIndex = CommenFunction_GetMaxIndexOfvector(distanceList);

			//����С���εĳ����б��е���󳤶ȶ�Ӧ�ĵ㼯��¼����
			allMaxPoints.push_back(allPoints[maxIndex]);

			testPoints.clear();
		}

		//���ˣ������߶γ��Ⱥ��߶ζ����Ƿ����������
		for (int i = 0; i < allMaxPoints.size(); i++)
		{
			//���㳤��
			double length = CommenFunction_DistanceOfTwoPoints(allMaxPoints[i][0], allMaxPoints[i][1]);

			if ((length >= 80))
			{
				line(blank1, allMaxPoints[i][0], allMaxPoints[i][1], Scalar(0, 0, 255), 2);
			}
		}

		return blank1;
	}

	//��������ͼ�������߶Σ����ػ����߶ε�ͼ��
	Mat BasicImgProc_GetAllTheLinesImgP(Mat srcImg, int minLength, cv::Rect mouthRect)
	{
		//hough�߼��
		vector<Vec4i> lines;//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
		HoughLinesP(srcImg, lines, 1, CV_PI / 180, minLength, minLength / 2, 5);

		//������ͼ�л��Ƴ�ÿ���߶�
		Mat LineImg = srcImg.clone();
		Mat rgbImg;
		cvtColor(LineImg, rgbImg, CV_GRAY2BGR);

		vector<Vec4i> filterlines;
		for (size_t i = 0; i < lines.size(); i++)
		{
			Vec4i I = lines[i];

			double length = CommenFunction_DistanceOfTwoPoints(Point(I[0], I[1]), Point(I[2], I[3]));

			//�ж��߶��Ƿ��ھ�����
			bool aa = clipLine(mouthRect, Point(I[0], I[1]), Point(I[2], I[3]));

			if ((length >= minLength) && (aa == true))
			{
				filterlines.push_back(I);
				line(rgbImg, Point(I[0], I[1]), Point(I[2], I[3]), Scalar(0, 0, 255), 1, CV_AA);
			}

		}



		////����б��
		//std::vector<double> slopeList;
		//for (int i = 0; i < filterlines.size(); i++)
		//{
		//	Vec4i line = filterlines[i];
		//	double tempK = ((double)line[3] - (double)line[1]) / ((double)line[2] - (double)line[0]);
		//	slopeList.push_back(tempK);
		//}

		//sort(slopeList.begin(), slopeList.end());
		//std::vector<int> subList;

		//for (int i = 0; i < slopeList.size() - 1; i++)
		//{
		//	if (slopeList[i + 1] - slopeList[i] <= 0.05)
		//	{
		//		subList.push_back(0);
		//	}
		//	else
		//	{
		//		subList.push_back(1);
		//	}
		//}

		//for (int i = 2; i <= 5; i++)
		//{
		//	Vec4i I = filterlines[i];
		//	//line(rgbImg, Point(I[0], I[1]), Point(I[2], I[3]), Scalar(0, 0, 255), 1, CV_AA);
		//}
		//imshow("1", rgbImg);

		/*CommenFunction_WhileWaitkey();*/

		return rgbImg;
	}

	//����Sobel�ݶ�ͼ
	Mat BasicImgProc_GetGradientImg(Mat srcImg)
	{
		Mat src, src_gray;
		Mat grad;
		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;

		if (srcImg.channels() == 3)
			cvtColor(srcImg, src_gray, CV_BGR2GRAY);
		else
			src_gray = srcImg.clone();

		/// ���� grad_x �� grad_y ����
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		/// �� X�����ݶ�
		Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		/// ��Y�����ݶ�
		Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		/// �ϲ��ݶ�(����)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		return grad;
	}

	//��ʾ����ͼ���ϣ����Է���
	void BasicImgProc_DisplayPoints(std::vector<Point> points)
	{
		Mat srcImg(1280, 720, CV_8UC1, Scalar(0));
		for (int i = 0; i < points.size(); i++)
		{
			circle(srcImg, points[i], 1, Scalar(255), 1);
		}
		imshow("debug", srcImg);
	}

	//�ݶ�ͼ���ֵ��
	Mat BasicImgProc_GetGradientBinaryImg(Mat srcImg)
	{
		//����������������ݶ�ͼ��
		Mat gradientImg = BasicImgProc_GetGradientImg(srcImg);

		//��������Ķ�ֵ�ݶ�ͼ����ͨȫ����ֵ��
		Mat dstImg1;
		threshold(gradientImg, dstImg1, 40, 255, THRESH_BINARY);

		//��������Ķ�ֵ�ݶ�ͼ��OSTUȫ����ֵ��
		Mat ostuImg;
		threshold(gradientImg, ostuImg, 0, 255, CV_THRESH_OTSU);

		//����
		Mat dstImg = ostuImg & dstImg1;

		return dstImg;
	}

	//ʹ��LSD���߶�
	std::vector<Vec4f> BasicImgProc_GetParallelLinesUsingLSD(Mat imgBinary, int minLength, int maxLength)
	{
		//��ƽ����
		Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
		std::vector<Vec4f> lines_std;

		// �߼��  
		ls->detect(imgBinary, lines_std);

		//����1������ƽ���߲���(����)
		std::vector<Vec4f> lines;
		for (int i = 0; i < lines_std.size(); i++)
		{
			Point p1(lines_std[i][0], lines_std[i][1]);
			Point p2(lines_std[i][2], lines_std[i][3]);

			double dis = CommenFunction_DistanceOfTwoPoints(p1, p2);

			if ((dis >= minLength) && (dis <= maxLength))
			{
				lines.push_back(lines_std[i]);
			}
		}

		return lines;
	}

	//��ͨ����
	int BasicImgProc_ConnectedAreaImgMark(const cv::Mat& _binImg, cv::Mat& _lableImg)
	{
		// connected component analysis(4-component)  
		// use seed filling algorithm  
		// 1. begin with a forgeground pixel and push its forground neighbors into a stack;  
		// 2. pop the pop pixel on the stack and label it with the same label until the stack is empty  
		//   
		//  forground pixel: _binImg(x,y)=1  
		//  background pixel: _binImg(x,y) = 0  


		if (_binImg.empty() || _binImg.type() != CV_8UC1)
		{
			return -1;
		}

		_lableImg.release();
		_binImg.convertTo(_lableImg, CV_32SC1);

		int label = 0; //start by 1  

		int rows = _binImg.rows;
		int cols = _binImg.cols;

		Mat mask(rows, cols, CV_8UC1);
		mask.setTo(0);
		int *lableptr;
		for (int i = 0; i < rows; i++)
		{
			int* data = _lableImg.ptr<int>(i);
			uchar *masKptr = mask.ptr<uchar>(i);
			for (int j = 0; j < cols; j++)
			{
				if (data[j] == 255 && mask.at<uchar>(i, j) != 1)
				{
					mask.at<uchar>(i, j) = 1;
					std::stack<std::pair<int, int>> neighborPixels;
					neighborPixels.push(std::pair<int, int>(i, j)); // pixel position: <i,j>  
					++label; //begin with a new label  
					while (!neighborPixels.empty())
					{
						//get the top pixel on the stack and label it with the same label  
						std::pair<int, int> curPixel = neighborPixels.top();
						int curY = curPixel.first;
						int curX = curPixel.second;
						_lableImg.at<int>(curY, curX) = label;

						//pop the top pixel  
						neighborPixels.pop();

						//push the 4-neighbors(foreground pixels)  

						if (curX - 1 >= 0)
						{
							if (_lableImg.at<int>(curY, curX - 1) == 255 && mask.at<uchar>(curY, curX - 1) != 1) //leftpixel  
							{
								neighborPixels.push(std::pair<int, int>(curY, curX - 1));
								mask.at<uchar>(curY, curX - 1) = 1;
							}
						}
						if (curX + 1 <= cols - 1)
						{
							if (_lableImg.at<int>(curY, curX + 1) == 255 && mask.at<uchar>(curY, curX + 1) != 1)
								// right pixel  
							{
								neighborPixels.push(std::pair<int, int>(curY, curX + 1));
								mask.at<uchar>(curY, curX + 1) = 1;
							}
						}
						if (curY - 1 >= 0)
						{
							if (_lableImg.at<int>(curY - 1, curX) == 255 && mask.at<uchar>(curY - 1, curX) != 1)
								// up pixel  
							{
								neighborPixels.push(std::pair<int, int>(curY - 1, curX));
								mask.at<uchar>(curY - 1, curX) = 1;
							}
						}
						if (curY + 1 <= rows - 1)
						{
							if (_lableImg.at<int>(curY + 1, curX) == 255 && mask.at<uchar>(curY + 1, curX) != 1)
								//down pixel  
							{
								neighborPixels.push(std::pair<int, int>(curY + 1, curX));
								mask.at<uchar>(curY + 1, curX) = 1;
							}
						}
					}
				}
			}
		}

		return label;
	}

	//������ͨ����ͼ�񣬵õ���ֵͼ�����е���ͨ��ͼ��
	std::map<int, Mat> BasicImgProc_GetAllConnectedAreaImgs(Mat binaryImg)
	{
		//����-ͼ��ӳ���б�
		std::map<int, Mat> index_ImgList;

		//��ͼ����ͨ����ͨ���Ƿ���newBinaryImg��
		Mat newBinaryImg;
		int labelNum = BasicImgProc_ConnectedAreaImgMark(binaryImg, newBinaryImg);

		//����������ͨ��

		int rows = newBinaryImg.rows;
		int cols = newBinaryImg.cols;

		for (int m = 1; m <= labelNum; m++)
		{
			Mat mask(newBinaryImg.size(), CV_8UC1, Scalar(0));
			for (int k = 0; k < rows; k++)
			{
				const int* data_src = (int*)newBinaryImg.ptr<int>(k);
				for (int j = 0; j < cols; j++)
				{
					int pixelValue = data_src[j];
					if (pixelValue == m)
					{
						mask.at<uchar>(k, j) = 255;
					}
				}
			}
			index_ImgList.insert(pair<int, Mat>(m - 1, mask));
		}

		return index_ImgList;
	}

	//ϸ��ͼ��
	Mat BasicImg_ProcGetThinImg(cv::Mat& src, int intera)
	{
		cv::Mat dst;

		//��ԭ�ز���ʱ��copy src��dst
		if (dst.data != src.data)
		{
			src.copyTo(dst);
		}

		int i, j, n;
		int width, height;
		width = src.cols - 1;
		//֮���Լ�1���Ƿ��㴦��8���򣬷�ֹԽ��
		height = src.rows - 1;
		int step = src.step;
		int  p2, p3, p4, p5, p6, p7, p8, p9;
		uchar* img;
		bool ifEnd;
		int A1;
		cv::Mat tmpimg;
		//n��ʾ��������
		for (n = 0; n<intera; n++)
		{
			dst.copyTo(tmpimg);
			ifEnd = false;
			img = tmpimg.data;
			for (i = 1; i < height; i++)
			{
				img += step;
				for (j = 1; j<width; j++)
				{
					uchar* p = img + j;
					A1 = 0;
					if (p[0] > 0)
					{
						if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01ģʽ
						{
							A1++;
						}
						if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01ģʽ
						{
							A1++;
						}
						if (p[1] == 0 && p[step + 1]>0) //p4,p5 01ģʽ
						{
							A1++;
						}
						if (p[step + 1] == 0 && p[step]>0) //p5,p6 01ģʽ
						{
							A1++;
						}
						if (p[step] == 0 && p[step - 1]>0) //p6,p7 01ģʽ
						{
							A1++;
						}
						if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01ģʽ
						{
							A1++;
						}
						if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01ģʽ
						{
							A1++;
						}
						if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01ģʽ
						{
							A1++;
						}
						p2 = p[-step]>0 ? 1 : 0;
						p3 = p[-step + 1]>0 ? 1 : 0;
						p4 = p[1]>0 ? 1 : 0;
						p5 = p[step + 1]>0 ? 1 : 0;
						p6 = p[step]>0 ? 1 : 0;
						p7 = p[step - 1]>0 ? 1 : 0;
						p8 = p[-1]>0 ? 1 : 0;
						p9 = p[-step - 1]>0 ? 1 : 0;
						if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)>1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)<7 && A1 == 1)
						{
							if ((p2 == 0 || p4 == 0 || p6 == 0) && (p4 == 0 || p6 == 0 || p8 == 0)) //p2*p4*p6=0 && p4*p6*p8==0
							{
								dst.at<uchar>(i, j) = 0; //����ɾ�����������õ�ǰ����Ϊ0
								ifEnd = true;
							}
						}
					}
				}
			}

			dst.copyTo(tmpimg);
			img = tmpimg.data;
			for (i = 1; i < height; i++)
			{
				img += step;
				for (j = 1; j<width; j++)
				{
					A1 = 0;
					uchar* p = img + j;
					if (p[0] > 0)
					{
						if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01ģʽ
						{
							A1++;
						}
						if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01ģʽ
						{
							A1++;
						}
						if (p[1] == 0 && p[step + 1]>0) //p4,p5 01ģʽ
						{
							A1++;
						}
						if (p[step + 1] == 0 && p[step]>0) //p5,p6 01ģʽ
						{
							A1++;
						}
						if (p[step] == 0 && p[step - 1]>0) //p6,p7 01ģʽ
						{
							A1++;
						}
						if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01ģʽ
						{
							A1++;
						}
						if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01ģʽ
						{
							A1++;
						}
						if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01ģʽ
						{
							A1++;
						}
						p2 = p[-step]>0 ? 1 : 0;
						p3 = p[-step + 1]>0 ? 1 : 0;
						p4 = p[1]>0 ? 1 : 0;
						p5 = p[step + 1]>0 ? 1 : 0;
						p6 = p[step]>0 ? 1 : 0;
						p7 = p[step - 1]>0 ? 1 : 0;
						p8 = p[-1]>0 ? 1 : 0;
						p9 = p[-step - 1]>0 ? 1 : 0;
						if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)>1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)<7 && A1 == 1)
						{
							if ((p2 == 0 || p4 == 0 || p8 == 0) && (p2 == 0 || p6 == 0 || p8 == 0)) //p2*p4*p8=0 && p2*p6*p8==0
							{
								dst.at<uchar>(i, j) = 0; //����ɾ�����������õ�ǰ����Ϊ0
								ifEnd = true;
							}
						}
					}
				}
			}

			//��������ӵ����Ѿ�û�п���ϸ���������ˣ����˳�����
			if (!ifEnd) break;
		}


		return dst;
	}

	//��ȡ��ֵͼ���еİ׵�
	std::vector<Point> BasicImg_GetWhitePointInBinaryImg(Mat binaryImg)
	{
		std::vector<Point> tempPoints;
		int rows = binaryImg.rows;
		int cols = binaryImg.cols;

		for (int k = 0; k < rows; k++)
		{
			vector<int> colsList;
			for (int j = 0; j < cols; j++)
			{
				//��ǰ��
				uchar t;
				t = binaryImg.at<uchar>(k, j);

				//λ��
				Point tempPoint(j, k);

				//������ǰ׵�
				if (t == 255)
				{
					tempPoints.push_back(tempPoint);
				}
			}
		}

		return tempPoints;
	}

	//���ݽǶ���ʱ����ת
	Mat BasicImg_ContrarotateImg(Mat srcImg, double angle)
	{
		cv::Mat src = srcImg.clone();
		cv::Point2f center(src.cols / 2, src.rows / 2);
		cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1);
		cv::Rect bbox = cv::RotatedRect(center, src.size(), angle).boundingRect();

		rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
		rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

		cv::Mat dst;
		cv::warpAffine(src, dst, rot, bbox.size());

		return dst;
	}

	//��ӽ�������(��ͨ��ͼ��)
	Mat BasicImg_AddSpiceAndAaltNoise(Mat img, double SNR)
	{
		Mat outImage;
		outImage.create(img.rows, img.cols, img.type());
		int SP = img.rows*img.cols;
		int NP = SP*(1 - SNR);
		outImage = img.clone();
		for (int i = 0; i<NP; i++)
		{
			int x = (int)(rand()*1.0 / RAND_MAX* (double)img.rows);
			int y = (int)(rand()*1.0 / RAND_MAX* (double)img.cols);
			int r = rand() % 2;
			if (r)
			{
				outImage.at<uchar>(x, y) = 0;
			}
			else
			{
				outImage.at<uchar>(x, y) = 255;
			}
		}

		return outImage;
	}

	//��Ӹ�˹��������ͨ��ͼ��
	Mat BasicImg_AddGassianNoise(Mat img, double mu, double sigma, int k, int PixcelMax, int PixcelMin)
	{
		Mat outImage;
		outImage.create(img.rows, img.cols, img.type());
		for (int x = 0; x<img.rows; x++)
		{
			for (int y = 0; y<img.cols; y++)
			{
				double temp = img.at<uchar>(x, y)
					+ k*generateGaussianNoise(mu, sigma);
				if (temp>PixcelMax)
					temp = PixcelMax;
				else if (temp<PixcelMin)
					temp = PixcelMin;
				outImage.at<uchar>(x, y) = temp;
			}
		}
		return outImage;
	}

#pragma endregion

#pragma region FaceProc

	//�õ�����������
	std::vector<Point> BasicImgProc_GetFacePoints(FaceFeature feature)
	{
		std::vector<Point> facePoints;
		for (int j = 0; j < 34; j++)
		{
			facePoints.push_back(Point(feature.PointX[j], feature.PointY[j]));
		}

		return facePoints;
	}

	//�õ���������
	cv::Rect BasicImgProc_GetSmokingAreaImg(std::vector<Point> facePoints)
	{
		Rect rect;

		int left = facePoints[0].x;
		int right = facePoints[6].x;

		int top;

		if (facePoints[0].y > facePoints[6].y)
		{
			top = facePoints[6].y;
		}
		else
		{
			top = facePoints[0].y;
		}

		int bottom = facePoints[3].y;

		rect.x = left;
		rect.y = top;
		rect.width = right - left;
		rect.height = bottom - top;


		return rect;
	}

	//�õ��첿����
	cv::Rect BasicImgProc_GetMouthRect(std::vector<Point> facePoints)
	{
		int left = facePoints[30].x;
		int right = facePoints[31].x;
		int top = facePoints[32].y;
		int bottom = facePoints[33].y;

		return Rect(left, top, right - left, bottom - top);
	}

	//����Ҷ�ͼ�����������㣬�õ���������
	Mat BasicImgProc_GetGrayContourFaceMask(Mat imgGray, std::vector<Point> facePoints)
	{
		int xCor[34], yCor[34];
		for (int i = 0; i < 34; i++)
		{
			xCor[i] = facePoints[i].x;
			yCor[i] = facePoints[i].y;
		}

		Point left, right;
		left.x = CommenFunction_GetMinValueOfArray(xCor);
		if (left.x <= 0)
		{
			left.x = 0;
		}

		left.y = CommenFunction_GetMinValueOfArray(yCor);
		right.x = CommenFunction_GetMaxValueOfArray(xCor);
		right.y = CommenFunction_GetMaxValueOfArray(yCor);

		if (right.y >= 720)
		{
			right.y = 720;
		}

		//������
		std::vector<Point> faceContour;
		faceContour.push_back(facePoints[0]);
		faceContour.push_back(facePoints[1]);
		faceContour.push_back(facePoints[2]);
		faceContour.push_back(facePoints[3]);
		faceContour.push_back(facePoints[4]);
		faceContour.push_back(facePoints[5]);
		faceContour.push_back(facePoints[6]);
		faceContour.push_back(facePoints[12]);
		faceContour.push_back(facePoints[11]);
		faceContour.push_back(facePoints[10]);
		faceContour.push_back(facePoints[9]);
		faceContour.push_back(facePoints[8]);
		faceContour.push_back(facePoints[7]);

		//��������
		std::vector<std::vector<Point>> faceContours;
		faceContours.push_back(faceContour);
		vector<Vec4i> hierarchy;

		//�Ҷ�ͼ������
		Mat mask(imgGray.size(), CV_8UC1, Scalar(0));
		drawContours(mask, faceContours, 0, Scalar(255), -1, 8, hierarchy);

		return mask;
	}

	//����Ҷ�ͼ�����������㣬�õ������Ҷ�����ͼ
	Mat BasicImgProc_GetGrayContourFace(Mat imgGray, std::vector<Point> facePoints)
	{
		int xCor[34], yCor[34];
		for (int i = 0; i < 34; i++)
		{
			xCor[i] = facePoints[i].x;
			yCor[i] = facePoints[i].y;
		}

		Point left, right;
		left.x = CommenFunction_GetMinValueOfArray(xCor);
		if (left.x <= 0)
		{
			left.x = 0;
		}

		left.y = CommenFunction_GetMinValueOfArray(yCor);
		right.x = CommenFunction_GetMaxValueOfArray(xCor);
		right.y = CommenFunction_GetMaxValueOfArray(yCor);

		if (right.y >= 720)
		{
			right.y = 720;
		}

		//������
		std::vector<Point> faceContour;
		faceContour.push_back(facePoints[0]);
		faceContour.push_back(facePoints[1]);
		faceContour.push_back(facePoints[2]);
		faceContour.push_back(facePoints[3]);
		faceContour.push_back(facePoints[4]);
		faceContour.push_back(facePoints[5]);
		faceContour.push_back(facePoints[6]);
		faceContour.push_back(facePoints[12]);
		faceContour.push_back(facePoints[11]);
		faceContour.push_back(facePoints[10]);
		faceContour.push_back(facePoints[9]);
		faceContour.push_back(facePoints[8]);
		faceContour.push_back(facePoints[7]);

		//��������
		std::vector<std::vector<Point>> faceContours;
		faceContours.push_back(faceContour);
		vector<Vec4i> hierarchy;

		//�Ҷ�ͼ������
		Mat mask(imgGray.size(), CV_8UC1, Scalar(0));
		drawContours(mask, faceContours, 0, Scalar(255), -1, 8, hierarchy);



		//ֻȡ����ͼ��
		Mat ImgFaceMask;
		bitwise_and(imgGray, mask, ImgFaceMask);

		//��ȡ����
		Mat smallFace = ImgFaceMask(Rect(left, right));

		return smallFace;
	}

	//����Ҷ�ͼ�����������㣬�õ���������Ҷ�����ͼ
	Mat BasicImgProc_GetGrayContourSmokingAreaOfFace(Mat imgGray, std::vector<Point> facePoints)
	{
		//������
		std::vector<Point> faceContour;
		faceContour.push_back(facePoints[0]);
		faceContour.push_back(facePoints[1]);
		faceContour.push_back(facePoints[2]);
		faceContour.push_back(facePoints[3]);
		faceContour.push_back(facePoints[4]);
		faceContour.push_back(facePoints[5]);
		faceContour.push_back(facePoints[6]);
		faceContour.push_back(facePoints[31]);
		faceContour.push_back(facePoints[30]);


		//��������
		std::vector<std::vector<Point>> faceContours;
		faceContours.push_back(faceContour);
		vector<Vec4i> hierarchy;

		//�Ҷ�ͼ������
		Mat mask(imgGray.size(), CV_8UC1);
		drawContours(mask, faceContours, 0, Scalar(255), -1, 8, hierarchy);

		//ֻȡ��������ͼ��
		Mat ImgFaceMask;
		bitwise_and(imgGray, mask, ImgFaceMask);

		int xCor[9], yCor[9];
		for (int i = 0; i < 9; i++)
		{
			xCor[i] = faceContour[i].x;
			yCor[i] = faceContour[i].y;
		}

		Point left, right;
		left.x = CommenFunction_GetMinValueOfArray(xCor);
		if (left.x <= 0)
		{
			left.x = 0;
		}

		left.y = CommenFunction_GetMinValueOfArray(yCor);
		right.x = CommenFunction_GetMaxValueOfArray(xCor);
		right.y = CommenFunction_GetMaxValueOfArray(yCor);

		if (right.y >= 720)
		{
			right.y = 720;
		}

		//��ȡ����
		Mat smallFace = ImgFaceMask(Rect(left, right));

		return smallFace;
	}

#pragma endregion

#pragma region FeatureExtraction

	//����Hog������ʹ��֮ǰ�ȳ߶ȹ�һ����64*128��
	vector<float> FeatureCalculate_GetHog(Mat src)
	{
		HOGDescriptor hog;//ʹ�õ���Ĭ�ϵ�hog����  
						  /*
						  HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8),
						  Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA(DEFAULT_WIN_SIGMA=-1),
						  double threshold_L2hys=0.2, bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)

						  Parameters:
						  win_size �C Detection window size. Align to block size and block stride.
						  block_size �C Block size in pixels. Align to cell size. Only (16,16) is supported for now.
						  block_stride �C Block stride. It must be a multiple of cell size.
						  cell_size �C Cell size. Only (8, 8) is supported for now.
						  nbins �C Number of bins. Only 9 bins per cell are supported for now.
						  win_sigma �C Gaussian smoothing window parameter.
						  threshold_L2hys �C L2-Hys normalization method shrinkage.
						  gamma_correction �C Flag to specify whether the gamma correction preprocessing is required or not.
						  nlevels �C Maximum number of detection window increases.
						  */
						  //����128*80��ͼƬ��blockstride = 8,15*9��block��2*2*9*15*9 = 4860  


						  //��ȡ�ߴ�
		hog.winSize = Size(src.cols, src.rows);

		//HOG��������  
		vector<float> des;

		//����
		hog.compute(src, des);//����hog����  
							  //Mat background = Mat::zeros(Size(width, height), CV_8UC1);//���ú�ɫ����ͼ����ΪҪ�ð�ɫ����hog����  
							  //Mat d = get_hogdescriptor_visual_image(background, des, hog.winSize, hog.cellSize, 3, 2.5);

		return des;
	}

	//����ͼ���Hog�������ӻ�ͼ��(scaleFactor=3, viz_factor=2.5)
	Mat FeatureCalculate_GetHogVisualImage(Mat src, int scaleFactor, double viz_factor)
	{
		Mat background = Mat::zeros(Size(src.cols, src.rows), CV_8UC1);

		HOGDescriptor hog;//ʹ�õ���Ĭ�ϵ�hog����  

		hog.winSize = Size(src.cols, src.rows);

		//HOG��������  
		vector<float> des;

		//����
		hog.compute(src, des);//����hog����  

		Mat visual_image = background.clone();//�����ӻ���ͼ���С  
		resize(src, visual_image, Size(src.cols*scaleFactor, src.rows*scaleFactor));

		int gradientBinSize = 9;
		// dividing 180�� into 9 bins, how large (in rad) is one bin?  
		float radRangeForOneBin = 3.14 / (float)gradientBinSize; //pi=3.14��Ӧ180��  

																 // prepare data structure: 9 orientation / gradient strenghts for each cell  
		int cells_in_x_dir = hog.winSize.width / hog.cellSize.width;//x�����ϵ�cell����  
		int cells_in_y_dir = hog.winSize.height / hog.cellSize.height;//y�����ϵ�cell����  
		int totalnrofcells = cells_in_x_dir * cells_in_y_dir;//cell���ܸ���  
															 //ע��˴���ά����Ķ����ʽ  
															 //int ***b;  
															 //int a[2][3][4];  
															 //int (*b)[3][4] = a;  
															 //gradientStrengths[cells_in_y_dir][cells_in_x_dir][9]  
		float*** gradientStrengths = new float**[cells_in_y_dir];
		int** cellUpdateCounter = new int*[cells_in_y_dir];
		for (int y = 0; y<cells_in_y_dir; y++)
		{
			gradientStrengths[y] = new float*[cells_in_x_dir];
			cellUpdateCounter[y] = new int[cells_in_x_dir];
			for (int x = 0; x<cells_in_x_dir; x++)
			{
				gradientStrengths[y][x] = new float[gradientBinSize];
				cellUpdateCounter[y][x] = 0;

				for (int bin = 0; bin<gradientBinSize; bin++)
					gradientStrengths[y][x][bin] = 0.0;//��ÿ��cell��9��bin��Ӧ���ݶ�ǿ�ȶ���ʼ��Ϊ0  
			}
		}

		// nr of blocks = nr of cells - 1  
		// since there is a new block on each cell (overlapping blocks!) but the last one  
		//�൱��blockstride = (8,8)  
		int blocks_in_x_dir = cells_in_x_dir - 1;
		int blocks_in_y_dir = cells_in_y_dir - 1;

		// compute gradient strengths per cell  
		int descriptorDataIdx = 0;
		int cellx = 0;
		int celly = 0;

		for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
		{
			for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
			{
				// 4 cells per block ...  
				for (int cellNr = 0; cellNr<4; cellNr++)
				{
					// compute corresponding cell nr  
					int cellx = blockx;
					int celly = blocky;
					if (cellNr == 1) celly++;
					if (cellNr == 2) cellx++;
					if (cellNr == 3)
					{
						cellx++;
						celly++;
					}

					for (int bin = 0; bin<gradientBinSize; bin++)
					{
						float gradientStrength = des[descriptorDataIdx];
						descriptorDataIdx++;

						gradientStrengths[celly][cellx][bin] += gradientStrength;//��ΪC�ǰ��д洢  

					} // for (all bins)  


					  // note: overlapping blocks lead to multiple updates of this sum!  
					  // we therefore keep track how often a cell was updated,  
					  // to compute average gradient strengths  
					cellUpdateCounter[celly][cellx]++;//����block֮�����ص�������Ҫ��¼��Щcell����μ�����  

				} // for (all cells)  


			} // for (all block x pos)  
		} // for (all block y pos)  


		  // compute average gradient strengths  
		for (int celly = 0; celly<cells_in_y_dir; celly++)
		{
			for (int cellx = 0; cellx<cells_in_x_dir; cellx++)
			{

				float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

				// compute average gradient strenghts for each gradient bin direction  
				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
				}
			}
		}


		cout << "winSize = " << hog.winSize << endl;
		cout << "cellSize = " << hog.cellSize << endl;
		cout << "blockSize = " << hog.cellSize * 2 << endl;
		cout << "blockNum = " << blocks_in_x_dir << "��" << blocks_in_y_dir << endl;
		cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

		// draw cells  
		for (int celly = 0; celly<cells_in_y_dir; celly++)
		{
			for (int cellx = 0; cellx<cells_in_x_dir; cellx++)
			{
				int drawX = cellx * hog.cellSize.width;
				int drawY = celly * hog.cellSize.height;

				int mx = drawX + hog.cellSize.width / 2;
				int my = drawY + hog.cellSize.height / 2;

				rectangle(visual_image,
					Point(drawX*scaleFactor, drawY*scaleFactor),
					Point((drawX + hog.cellSize.width)*scaleFactor,
					(drawY + hog.cellSize.height)*scaleFactor),
					CV_RGB(0, 0, 0),//cell���ߵ���ɫ  
					1);

				// draw in each cell all 9 gradient strengths  
				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float currentGradStrength = gradientStrengths[celly][cellx][bin];

					// no line to draw?  
					if (currentGradStrength == 0)
						continue;

					float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;//ȡÿ��bin����м�ֵ����10��,30��,...,170��.  

					float dirVecX = cos(currRad);
					float dirVecY = sin(currRad);
					float maxVecLen = hog.cellSize.width / 2;
					float scale = viz_factor; // just a visual_imagealization scale,  
											  // to see the lines better  

											  // compute line coordinates  
					float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
					float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
					float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
					float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

					// draw gradient visual_imagealization  
					line(visual_image,
						Point(x1*scaleFactor, y1*scaleFactor),
						Point(x2*scaleFactor, y2*scaleFactor),
						CV_RGB(255, 255, 255),//HOG���ӻ���cell����ɫ  
						1);

				} // for (all bins)  

			} // for (cellx)  
		} // for (celly)  


		  // don't forget to free memory allocated by helper data structures!  
		for (int y = 0; y<cells_in_y_dir; y++)
		{
			for (int x = 0; x<cells_in_x_dir; x++)
			{
				delete[] gradientStrengths[y][x];
			}
			delete[] gradientStrengths[y];
			delete[] cellUpdateCounter[y];
		}
		delete[] gradientStrengths;
		delete[] cellUpdateCounter;

		return visual_image;//�������յ�HOG���ӻ�ͼ��  

	}

#pragma endregion

#pragma region SimpleFunction

	//������˹�����
	double generateGaussianNoise(double mu, double sigma)
	{
		static double V1, V2, S;
		static int phase = 0;
		double X;
		double U1, U2;
		if (phase == 0) {
			do {
				U1 = (double)rand() / RAND_MAX;
				U2 = (double)rand() / RAND_MAX;

				V1 = 2 * U1 - 1;
				V2 = 2 * U2 - 1;
				S = V1 * V1 + V2 * V2;
			} while (S >= 1 || S == 0);

			X = V1 * sqrt(-2 * log(S) / S);
		}
		else {
			X = V2 * sqrt(-2 * log(S) / S);
		}
		phase = 1 - phase;
		return mu + sigma*X;
	}

#pragma endregion

}
