#pragma once
#pragma once
#pragma once
#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\imgproc.hpp>

using namespace cv;
using namespace std;

//ÀàÉùÃ÷
class  SafebeltDetect
{

public:
	SafebeltDetect::SafebeltDetect();
	Mat SafebeltDetect::GetBinaryImg();

	bool SafebeltDetect::DetectIfSafebelt(Mat img, std::vector<Point> faceLandmarkPoints);


	Mat SafebeltDetect::GetmidResultSafebeltImg();
	Mat SafebeltDetect::GetmidResultSafebeltImg1();
	Mat SafebeltDetect::GetmidResultSafebeltImg2();
	Mat SafebeltDetect::GetfinalResultImg();

	//std::vector<Vec3f> SomkingDetect::getcircle();
};

