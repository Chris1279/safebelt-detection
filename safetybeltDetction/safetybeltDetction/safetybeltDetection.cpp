#include <opencv2\opencv.hpp>
#include <opencv2\imgproc.hpp>
#include "safebeltDetection.h"

using namespace cv;
using namespace std;

#pragma region �궨�� 

#define DLIB34
//#define SELF21

//ͼ���һ���߶�
#define RESIZED_WIDTH			160
#define RESIZED_HEIGHT			80

//���������㶨��
#ifdef	DLIB34 

#define FACE_LEFT				0
#define FACE_RIGHT				6
#define FACE_UP					16
#define FACE_DOWN				3

#define MOUTH_LEFT				30
#define MOUTH_RIGHT				31
#define MOUTH_UP				32
#define MOUTH_DOWN				33

#define EYEBROWS                13

#define NOSE_CENTER              16
#endif 

#ifdef	SELF21 

#define FACE_LEFT				0
#define FACE_RIGHT				2
#define FACE_UP					15
#define FACE_DOWN				1

#define MOUTH_LEFT				17
#define MOUTH_RIGHT				18
#define MOUTH_UP				19
#define MOUTH_DOWN				20

#endif 

//�ж�Ϊƽ���߽ǶȲ�ֵ
#define PARALLE_THETA            10

//�������ӵ����Ƕȱ仯
#define CONNECT_THETA            10
#pragma endregion

#pragma region ȫ�ֱ��� 

//ԭʼͼ��
Mat _srcImg;

//����������
std::vector<Point> _faceLandmarkPoints;

//��ֵ��ͼ��
Mat _binaryImg;

//��ȫ���м���ͼ��
Mat _midResultSafebeltImg;
Mat _midResultSafebeltImg1;
Mat _midResultSafebeltImg2;

//���ս��
Mat _finalResultImg;

Size imgSize;

//��ȫ������������
cv::Rect _detectSafebeltAreaRectangle;

//����
Rect faceRect;

//�߶ȱ�׼
int standardSafebelt;

//���ȵ����޺�����
double minLength;
double maxLength;

//ƽ���߼����ֵ
double maxWidth;
double minWidth;

//ƽ���߽Ƕ���ֵ
double maxTheta;
double minTheta;

//�������ӵ���С�˵����
double connectDistance;

//���Ӻ󳤶�����
double minLongLength;

//��ȫ���Ŵ���
int scaleArea1 = 3;

#pragma endregion

#pragma region ˽�к���

//��Ƶ����ȫ��
int drawSafebelLinne(std::vector<Point> facePoints, Vec4f v) {
	//int d = facePoints[FACE_RIGHT].x - facePoints[FACE_LEFT].x;
	//int h = imgSize.height - facePoints[MOUTH_DOWN].y;
	int x1 = v[0] / scaleArea1 + _detectSafebeltAreaRectangle.x;
	int y1 = v[1] / scaleArea1 + _detectSafebeltAreaRectangle.y;
	int x2 = v[2] / scaleArea1 + _detectSafebeltAreaRectangle.x;
	int y2 = v[3] / scaleArea1 + _detectSafebeltAreaRectangle.y;
	line(_finalResultImg, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 2);
	return 0;
}

//�����������
double DistanceOfTwoPoints(Point p1, Point p2)
{
	double sum = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
	return sqrt(sum);
}

//ƽ���߽���
double intersectionPercent(Vec4f line1, Vec4f line2, double theta) {
	int a, b, c, d;
	if (fabs(theta) > 45) {
		a = min(line1[1], line1[3]);
		b = max(line1[1], line1[3]);
		c = min(line2[1], line2[3]);
		d = max(line2[1], line2[3]);
	}
	else {
		a = min(line1[0], line1[2]);
		b = max(line1[0], line1[2]);
		c = min(line2[0], line2[2]);
		d = max(line2[0], line2[2]);
	}

	double pertage;
	if (a < c) {
		if (d < b)
			pertage = (d - c) / 50.0;
		else
			if (c < b)
				pertage = (b - c) / 50.0;
			else
				pertage = 0;
	}
	else {
		if (b < d)
			pertage = (b - a) / 50.0;
		else
			if (d > a)
				pertage = (d - a) / 50.0;
			else
				pertage = 0;
	}
	return pertage;
}

//���ھ�����ı���
double lineRect(Rect lineRect, Vec4f line, double theta) {
	double inRate, inNum = 0;
	int leftX, rightX, upY, downY;
	Point linePoint;
	vector<Point> mouthRectContour = { Point(lineRect.x,lineRect.y),Point(lineRect.x + lineRect.width,lineRect.y) ,Point(lineRect.x + lineRect.width,lineRect.y + lineRect.height) ,Point(lineRect.x,lineRect.y + lineRect.height) };
	double k = (double(line[3] - line[1])) / (double(line[2] - line[0]));
	if (fabs(k) < 1) {
		if (line[0] < line[2]) {
			leftX = line[0];
			rightX = line[2];
		}
		else {
			leftX = line[2];
			rightX = line[0];
		}
		for (int i = leftX; i < rightX; i++) {
			int y = k*(i - line[0]) + line[1];
			linePoint = Point(i, y);
			if (pointPolygonTest(mouthRectContour, linePoint, false) == 1)
				inNum++;
		}
		inRate = inNum / (rightX - leftX);
	}
	else {
		if (line[1] < line[3]) {
			upY = line[1];
			downY = line[3];
		}
		else {
			upY = line[3];
			downY = line[1];
		}
		for (int j = upY; j < downY; j++) {
			if (k == 0)
				linePoint = Point(line[0], j);
			else
				linePoint = Point((j - line[1]) / k + line[0], j);
			if (pointPolygonTest(mouthRectContour, linePoint, false) == 1)
				inNum++;
		}
		inRate = inNum / (downY - upY);
	}
	return inRate;
}


//����ƽ�����ӳ����Ƿ񾭹��첿
bool ParallelLinesInmouth(Rect lineRect1,Vec4f line, double theta) {

	//Rect rect = boundingRect(mouthContour);
	//mouthRect=Rect(rect.x, rect.y- minWidth, rect.width, rect.height+2* minWidth);
	Point p5(lineRect1.x, lineRect1.y);
	Point p6(lineRect1.x + lineRect1.width, lineRect1.y + lineRect1.height);

	rectangle(_midResultSafebeltImg, p5, p6, Scalar(255, 0, 0));
	rectangle(_midResultSafebeltImg, p5, p6, Scalar(255, 0, 0));
	int mouthHeight = (_faceLandmarkPoints[MOUTH_RIGHT].y + _faceLandmarkPoints[MOUTH_LEFT].y - 2 * _detectSafebeltAreaRectangle.y) / 2 * scaleArea1;
	if (lineRect(lineRect1, line, theta) > 0)
		return 1;
	else {
		Point nearPoint;
		if (abs(line[1] - mouthHeight) < abs(line[3] - mouthHeight))
			nearPoint = Point(line[0], line[1]);
		else
			nearPoint = Point(line[2], line[3]);
		for (int i = lineRect1.x; i < lineRect1.x + lineRect1.width; i++) {
			for (int j = lineRect1.y; j < lineRect1.y + lineRect1.height; j++) {
				Point p(i, j);
				//if (pointPolygonTest(mouthContour, p, false) == 1) 
				{
					//if (pointLine(p, line))
					//return 1;
					double k = (double(j - nearPoint.y)) / (double(i - nearPoint.x));
					double theta1 = atan(k) * 180 / CV_PI;
					double distance = DistanceOfTwoPoints(p, nearPoint);
					if ((abs(theta1 - theta) <3) && distance < 40)
						return 1;
				}
			}
		}
	}

	return 0;

}

//ʹ��LSD���߶�
std::vector<Vec4f> GetParallelLinesUsingLSD(Mat imgBinary, int minL, int maxL)
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
		line(_binaryImg, p1, p2, Scalar(0, 255, 0), 2);
		double dis = DistanceOfTwoPoints(p1, p2);
		if ((dis >= minL) && (dis <= maxL))
		{
			lines.push_back(lines_std[i]);
		}

	}
	//namedWindow("4", 0);
	//imshow("4", imgBinary);
	//waitKey(1);
	return lines;
}

//����ƽ���߼�ľ���
double GetDisOfParallelLines(Vec4f line0, Vec4f line1)
{
	CvPoint midPoint = cvPoint((line0[0] + line0[2]) / 2, (line0[1] + line0[3]) / 2); // �е�  
	double x_dis = line1[0] - line1[2];
	if (x_dis == 0.0) return fabs((double)(midPoint.x - line0[0])); // ���line1 ��ֱx��  

	double a = (line1[1] - line1[3]) / x_dis;
	double b = line1[1] - (line1[0] * a);
	return fabs(a * midPoint.x - midPoint.y + b) / sqrt(a * a + 1);
}

//����ֱ��б��
double thetaPoint(Point p1, Point p2) {
	double theta;
	if (p2.x != p1.x) {
		double k = (double(p2.y - p1.y)) / (double(p2.x - p1.x));
		theta = atan(k) * 180 / CV_PI;
	}
	else
		theta = 90;
	return theta;
}

//���ƽ���߶�
std::vector<Vec4f> GetAllParallelLines(std::vector<Vec4f> lines, std::vector<double> &thetaList1) {
	//float rho = lines[k][0], theta = lines[k][1];
	//Point pt1, pt2;
	//double a = cos(theta), b = sin(theta);
	//double x0 = a*rho, y0 = b*rho;
	//pt1.x = cvRound(x0 + 1000 * (-b));
	//pt1.y = cvRound(y0 + 1000 * (a));
	//pt2.x = cvRound(x0 - 1000 * (-b));
	//pt2.y = cvRound(y0 - 1000 * (a));


	//double line2length = DistanceOfTwoPoints(Point(line2[0], line2[1]), Point(line2[2], line2[3]));

	//����ƽ���߲���(��б��)
	std::vector<double> thetaList;
	for (int i = 0; i < lines.size(); i++)
	{
		Point p1(lines[i][0], lines[i][1]);
		Point p2(lines[i][2], lines[i][3]);
		double theta;
		if (p2.x != p1.x) {
			double k = (double(p2.y - p1.y)) / (double(p2.x - p1.x));
			theta = atan(k) * 180 / CV_PI;
		}
		else
			theta = 90;
		thetaList.push_back(theta);
	}

	//�ҳ����е�ƽ���߶�
	std::vector<Vec4f> ParallelLinesList;
	if (lines.size() >= 2) {
		for (int k = 0; k < lines.size() - 1; k++)
		{
			//�߶γ���ɾѡ
			double line1length = DistanceOfTwoPoints(Point(lines[k][0], lines[k][1]), Point(lines[k][2], lines[k][3]));
			if (line1length < minLongLength)
				continue;
			for (int j = k + 1; j < lines.size(); j++) {
				double theta_sub = abs(thetaList[k] - thetaList[j]);
				if (theta_sub <= 10) {
					ParallelLinesList.push_back(lines[k]);
					ParallelLinesList.push_back(lines[j]);
					thetaList1.push_back(thetaList[k]);
					thetaList1.push_back(thetaList[j]);
				}
			}
		}
	}
	return ParallelLinesList;
}

//�߶�����
std::vector<Vec4f> GetLongLines(std::vector<Vec4f> lines, std::vector<double> thetaList, int tolerateValue) {
	
	std::vector<Vec4f> linesLong;
	for (int i = 0; i < lines.size() - 1; i++) {
		for (int j = i + 1; j < lines.size(); j++) {
			//int j = i + 1;
			if (fabs(thetaList[i] - thetaList[j]) < 5) {
				vector<Point> pointList;
				pointList.push_back(Point(lines[i][0], lines[i][1]));
				pointList.push_back(Point(lines[i][2], lines[i][3]));
				pointList.push_back(Point(lines[j][0], lines[j][1]));
				pointList.push_back(Point(lines[j][2], lines[j][3]));
				double disPoint[4];
				disPoint[0] = DistanceOfTwoPoints(pointList[0], pointList[2]);
				disPoint[1] = DistanceOfTwoPoints(pointList[0], pointList[3]);
				disPoint[2] = DistanceOfTwoPoints(pointList[1], pointList[2]);
				disPoint[3] = DistanceOfTwoPoints(pointList[1], pointList[3]);
				double theta[4];
				theta[0] = thetaPoint(pointList[0], pointList[2]);
				theta[1] = thetaPoint(pointList[0], pointList[3]);
				theta[2] = thetaPoint(pointList[1], pointList[2]);
				theta[3] = thetaPoint(pointList[1], pointList[3]);
				int minNum = min_element(disPoint, end(disPoint)) - disPoint;
				int maxNum = max_element(disPoint, end(disPoint)) - disPoint;
				double mindisTheta = theta[minNum] - (thetaList[i] + thetaList[i]) / 2;
				double maxdisTheta = theta[maxNum] - (thetaList[i] + thetaList[i]) / 2;
				// && (fabs(mindisTheta) < 20)
				if ((disPoint[minNum] < tolerateValue) && (fabs(maxdisTheta) < 3)) {
					//double mindisPoint = *min_element(disPoint, end(disPoint));

					//if (fabs(theta[maxNum] - (thetaList[i] + thetaList[i + 1]) / 2) < 3){
					Vec4f lineSingle;
					switch (maxNum)
					{
					case 0:
						lineSingle = { lines[i][0], lines[i][1],lines[j][0], lines[j][1] }; break;
					case 1:
						lineSingle = { lines[i][0], lines[i][1],lines[j][2], lines[j][3] }; break;
					case 2:
						lineSingle = { lines[i][2], lines[i][3],lines[j][0], lines[j][1] }; break;
					case 3:
						lineSingle = { lines[i][2], lines[i][3],lines[j][2], lines[j][3] }; break;
					default:
						break;
					}
					//linesLong.erase(linesLong.begin() + i);
					//linesLong.erase(linesLong.begin() + j);
					lines[i] = lineSingle;
					lines[j] = lineSingle;
					linesLong.push_back(lineSingle);
				}
				//else {
				//	linesLong.push_back(lines[i]);
				//	linesLong.push_back(lines[i]);
				//}
			}
		}

	}
	return lines;
}

//��ȡ��ȫ���������
int GetRectOfSafebeltAreaImg(std::vector<Point> facePoints, Size imgSize) {
	standardSafebelt = facePoints[NOSE_CENTER].y - facePoints[EYEBROWS].y;
	_detectSafebeltAreaRectangle.x = facePoints[MOUTH_DOWN].x - 5 * standardSafebelt;
	//_detectSafebeltAreaRectangle.y = facePoints[MOUTH_DOWN].y + 2 * standardSafebelt;
	//_detectSafebeltAreaRectangle.y = facePoints[FACE_DOWN].y ;
	_detectSafebeltAreaRectangle.y = facePoints[EYEBROWS].y;
	_detectSafebeltAreaRectangle.width = 8 * standardSafebelt;
	_detectSafebeltAreaRectangle.height = 5 * standardSafebelt;

	if (_detectSafebeltAreaRectangle.x < 0)
		_detectSafebeltAreaRectangle.x = 0;

	if (_detectSafebeltAreaRectangle.y > imgSize.height)
		_detectSafebeltAreaRectangle.y = facePoints[MOUTH_DOWN].y;

	if (_detectSafebeltAreaRectangle.x + _detectSafebeltAreaRectangle.width > imgSize.width)
		_detectSafebeltAreaRectangle.width = imgSize.width - _detectSafebeltAreaRectangle.x;

	//if (_detectSafebeltAreaRectangle.y + _detectSafebeltAreaRectangle.height > imgSize.height)
	_detectSafebeltAreaRectangle.height = imgSize.height - _detectSafebeltAreaRectangle.y;
	return 0;
}

//����
int GetRectOfSafebeltFace(std::vector<Point> facePoints, Size imgSize) {

	faceRect.x = (facePoints[FACE_LEFT].x - _detectSafebeltAreaRectangle.x)*scaleArea1;
	faceRect.y = (facePoints[EYEBROWS].y - _detectSafebeltAreaRectangle.y)*scaleArea1;
	faceRect.width = (facePoints[FACE_RIGHT].x - facePoints[FACE_LEFT].x)*scaleArea1;
	faceRect.height = (facePoints[FACE_DOWN].y - facePoints[EYEBROWS].y)*scaleArea1;
	if (faceRect.x < 0)
		faceRect.x = 0;

	if (faceRect.y <0)
		faceRect.y = 0;

	if (faceRect.x + faceRect.width > _detectSafebeltAreaRectangle.width*scaleArea1)
		faceRect.width = _detectSafebeltAreaRectangle.width*scaleArea1 - faceRect.x;

	if (faceRect.y + faceRect.height > _detectSafebeltAreaRectangle.height*scaleArea1)
		faceRect.height = _detectSafebeltAreaRectangle.height*scaleArea1 - faceRect.y;
}
//��ⰲȫ��
bool DetectSafebelt(std::vector<Point> facePoints, Mat img) {

	GetRectOfSafebeltAreaImg(facePoints, imgSize);

	if (_detectSafebeltAreaRectangle.height == 0)
		return 0;

	Mat safebeltAreaImg = img(_detectSafebeltAreaRectangle);
	Mat safebeltAreaImg5;
	resize(safebeltAreaImg, safebeltAreaImg5, Size(0, 0), scaleArea1, scaleArea1);
	cvtColor(safebeltAreaImg5, _midResultSafebeltImg, CV_GRAY2RGB);
	_midResultSafebeltImg1 = _midResultSafebeltImg.clone();
	_midResultSafebeltImg2 = _midResultSafebeltImg.clone();

	//����ֱ��
	std::vector<Vec4f> lines1 = GetParallelLinesUsingLSD(safebeltAreaImg5, minLength, maxLength);
	Mat lineMat(safebeltAreaImg5.size(), CV_8UC1, Scalar(0));
	for (int k = 0; k<lines1.size(); k++)
		line(lineMat, Point(lines1[k][0], lines1[k][1]), Point(lines1[k][2], lines1[k][3]), Scalar(255), 2);
	imshow("����ֱ��", lineMat);
	cvWaitKey(1);

	//for (int k = 0; k<lines1.size(); k++)
	//	line(_midResultSafebeltImg, Point(lines1[k][0], lines1[k][1]), Point(lines1[k][2], lines1[k][3]), Scalar(0, 0, 255), 2);

	//����ƽ���߲���(��б��)
	std::vector<double> thetaListAll;
	for (int i = 0; i < lines1.size(); i++)
	{
		Point p1(lines1[i][0], lines1[i][1]);
		Point p2(lines1[i][2], lines1[i][3]);
		double theta;
		if (p2.x != p1.x) {
			double k = (double(p2.y - p1.y)) / (double(p2.x - p1.x));
			theta = atan(k) * 180 / CV_PI;
		}
		else
			theta = 90;
		thetaListAll.push_back(theta);
	}

	//�����߶�
	std::vector<Vec4f> lineslong = GetLongLines(lines1, thetaListAll, connectDistance);
	Mat lineMat1(safebeltAreaImg5.size(), CV_8UC1, Scalar(0));
	for (int k = 0; k<lines1.size(); k++)
		line(lineMat1, Point(lineslong[k][0], lineslong[k][1]), Point(lineslong[k][2], lineslong[k][3]), Scalar(255), 2);
	imshow("�������Ӻ��ֱ��", lineMat1);
	cvWaitKey(1);
		
	//����ƽ����
	std::vector<double> thetaList;
	std::vector<Vec4f> ParallelLines = GetAllParallelLines(lineslong, thetaList);
	if (ParallelLines.size() < 2)
		return 0;
	for (int k = 0; k<ParallelLines.size(); k++)
		line(_midResultSafebeltImg, Point(ParallelLines[k][0], ParallelLines[k][1]), Point(ParallelLines[k][2], ParallelLines[k][3]), Scalar(0, 0, 255), 2);

	double maxper = 0;
	int num;
	for (int i = 0; i < ParallelLines.size() - 1; i += 2) {
		Vec4f line1 = ParallelLines[i];
		Vec4f line2 = ParallelLines[i + 1];
		double line1length = DistanceOfTwoPoints(Point(line1[0], line1[1]), Point(line1[2], line1[3]));
		double line2length = DistanceOfTwoPoints(Point(line2[0], line2[1]), Point(line2[2], line2[3]));
		//if (line1length < minWidth || line2length < minWidth)
		//continue;
		double dis = GetDisOfParallelLines(line1, line2);

		//��ȫ���ĽǶ���һ����Χ��
		if (thetaList[i] > minTheta && thetaList[i] < maxTheta)
		{
			//ƽ���߾���
			//if (lineRect(faceRect, line1, thetaList[i]) > 0.7 || lineRect(faceRect, line1, thetaList[i]) > 0.7)
			//	continue;
			if (dis > minWidth && dis < maxWidth) {
				cout << thetaList[i] << endl;
				line(_midResultSafebeltImg1, Point(line1[0], line1[1]), Point(line1[2], line1[3]), Scalar(0, 0, 255), 2);
				line(_midResultSafebeltImg1, Point(line2[0], line2[1]), Point(line2[2], line2[3]), Scalar(0, 0, 255), 2);
				//ƽ���߽�������
				double mixper = intersectionPercent(line1, line2, thetaList[i]);
				if (mixper > maxper) {
					maxper = mixper;
					num = i;
				}
				cout << dis << " ";
			}
		}
	}
	if (maxper > 0) {
		line(_midResultSafebeltImg2, Point(ParallelLines[num][0], ParallelLines[num][1]), Point(ParallelLines[num][2], ParallelLines[num][3]), Scalar(0, 0, 255), 2);
		line(_midResultSafebeltImg2, Point(ParallelLines[num + 1][0], ParallelLines[num + 1][1]), Point(ParallelLines[num + 1][2], ParallelLines[num + 1][3]), Scalar(0, 0, 255), 2);
		drawSafebelLinne(facePoints, ParallelLines[num]);
		drawSafebelLinne(facePoints, ParallelLines[num + 1]);
		//cout << thetaList[num] << endl;
		return 1;
	}
	return 0;
}

bool DODetectIfSafebelt(std::vector<Point> facePoints, Mat img) {
	// ȫ�ֱ���
	//�߶ȱ�׼
	standardSafebelt = facePoints[NOSE_CENTER].y - facePoints[EYEBROWS].y;

	//���ȵ����޺�����
	minLength = 0.2*standardSafebelt*scaleArea1;
	maxLength = 10 * standardSafebelt*scaleArea1;

	//ƽ���߼����ֵ
	minWidth = 0.7*standardSafebelt*scaleArea1;
	maxWidth = 1.2*standardSafebelt*scaleArea1;

	//��ȫ���Ƕ���ֵ	
	minTheta = 40;
	maxTheta = 80;

	//�������ӵ���С�˵����
	connectDistance = 0.7*standardSafebelt* scaleArea1;

	//���Ӻ󳤶�����
	minLongLength = 1.2*standardSafebelt*scaleArea1;

	bool SafebeltIshere = false;
	SafebeltIshere = DetectSafebelt(facePoints, img);
	return SafebeltIshere;
}
#pragma endregion

#pragma region ���캯��

//���캯��
SafebeltDetect::SafebeltDetect()
{
	
}

#pragma endregion

#pragma region ȫ�ֺ���
//��ⰲȫ��
bool SafebeltDetect::DetectIfSafebelt(Mat img, std::vector<Point> faceLandmarkPoints) {
	Mat imgGray;
	cvtColor(img, imgGray, CV_BGR2GRAY);

	_finalResultImg = img.clone();
	_srcImg = imgGray;
	_faceLandmarkPoints = faceLandmarkPoints;
	imgSize = _srcImg.size();
	//drawFacePoint(imgGray, _faceLandmarkPoints);
		
	bool ifSafebelt = false;
	ifSafebelt = DODetectIfSafebelt(_faceLandmarkPoints, _srcImg);
	return ifSafebelt;
}

//�õ���ֵ��ͼ��
Mat SafebeltDetect::GetBinaryImg()
{
	return _binaryImg;
}

//��ȫ���м���
Mat SafebeltDetect::GetmidResultSafebeltImg()
{
	return _midResultSafebeltImg;
}
Mat SafebeltDetect::GetmidResultSafebeltImg1()
{
	return _midResultSafebeltImg1;
}
Mat SafebeltDetect::GetmidResultSafebeltImg2()
{
	return _midResultSafebeltImg2;
}

//�õ����ս��ͼ��
Mat SafebeltDetect::GetfinalResultImg()
{
	return _finalResultImg;
}

#pragma endregion
