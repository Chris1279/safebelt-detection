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

//�㷨���ģ��
void SafebeltDetectModule(VideoCapture capture)
{
	//��ʼ��
	std::vector<Point> facePoints;
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor shaperModel;
	dlib::deserialize("sp.dat") >> shaperModel;

	//���̼��ģ��
	SafebeltDetect temp;

	int index = 0;
	//����
	long currentFrame = 1;
	//��ȡͼ��
	while (1)
	{
		//���ͼ��
		Mat srcImg;
		capture >> srcImg;
		Mat finalImg = srcImg.clone();
		//���Ϊ���˳�
		if (srcImg.empty())
		{
			break;
		}
		else
		{
			//�������
			Mat imgGray;
			cvtColor(srcImg, imgGray, CV_BGR2GRAY);
			cv_image<unsigned char> cimg(imgGray);
			std::vector<dlib::rectangle> faces = detector(cimg);

			//��⵽����
			if (faces.size() > 0)
			{
				//---------------------------�õ�landmark�㼯
				std::vector<full_object_detection> shapes;

				for (unsigned long i = 0; i < faces.size(); ++i)
					shapes.push_back(shaperModel(cimg, faces[i]));

				for (unsigned long i = 0; i <= 33; ++i)
				{
					int x = shapes[0].part(i).x();
					int y = shapes[0].part(i).y();
					facePoints.push_back(Point(x, y));
				}

				//------------------------------------------------------------------------���������Ƿ�ϵ��ȫ��	
				bool ifSafebelt = temp.DetectIfSafebelt(srcImg, facePoints);

				//��ʾ��ֵ��ͼ��
				Mat binaryIMg = temp.GetBinaryImg();

				//��ȫ���м���
				Mat midSafebeltImg = temp.GetmidResultSafebeltImg();
				Mat midSafebeltImg1 = temp.GetmidResultSafebeltImg1();
				Mat midSafebeltImg2 = temp.GetmidResultSafebeltImg2();

				Mat midSafebeltshowImg, midSafebeltshowImg1, midSafebeltshowImg2;
				resize(midSafebeltImg, midSafebeltshowImg, Size(0, 0), 0.5, 0.5);
				resize(midSafebeltImg1, midSafebeltshowImg1, Size(0, 0), 0.5, 0.5);
				resize(midSafebeltImg2, midSafebeltshowImg2, Size(0, 0), 0.5, 0.5);

				finalImg = temp.GetfinalResultImg();

				imshow("�м���������ƽ���߰�ȫ��", midSafebeltshowImg);
				imshow("�м���������ɾ��", midSafebeltshowImg1);
				imshow("�м�������󽻼�", midSafebeltshowImg2);

				//����ÿһ֡�ĳ�������ͼ��
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
					////����ͼ��
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
			//���ķŴ��屶����Сʮ��
			Mat imgShow;
			resize(finalImg, imgShow, Size(0, 0), 1, 1);
			imshow("1", imgShow);

			//------------------------------------------------------------------------end

			facePoints.clear();
		}
		cv::waitKey(1);
	}
}

//ʹ�õ�����Ƶ�����㷨(�����ļ���)
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

//������
void main()
{
	//�����ļ����µ�������Ƶ
	TestAllTheVideos("G:\\testVideo\\safebelt\\³HR9333\\");

	////��������ͷ����
	//SmokingDetectModule(VideoCapture(0));	

	//TestFaceAlign();
}
