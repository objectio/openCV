#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int hue, saturation, value;
Mat src; 	Mat srcHSV;
Mat srcThresholded;

void on_trackbar(int pos, void*) {
	inRange(srcHSV, Scalar(0, 0, value), Scalar(195, 225, 235), srcThresholded);
	imshow("Thresholded Image", srcThresholded);
}

int main() {
	src = imread("image.jpg", IMREAD_COLOR);
	if (src.empty()) {
		cout << "������ ���� �� ����" << endl;
		return (-1);
	}

	cvtColor(src, srcHSV, COLOR_BGR2HSV);
	namedWindow("src", 1);
	imshow("src", src);

	createTrackbar("H", "src", &hue, 255, on_trackbar);
	createTrackbar("S", "src", &saturation, 255, on_trackbar);
	createTrackbar("V", "src", &value, 255, on_trackbar);

	// ���� 2
	Mat img = imread("selfie.jpg", IMREAD_COLOR);
	Mat img2 = imread("back.jpg", IMREAD_COLOR);

	Mat converted;
	cvtColor(img, converted, COLOR_BGR2HSV);
	Mat greenScreen = converted.clone();
	inRange(converted, Scalar(170, 210, 96),
		Scalar(179, 255, 255), greenScreen);

	Mat dst, dst1, inverted;
	bitwise_not(greenScreen, inverted);
	bitwise_and(img, img, dst, inverted);
	bitwise_or(dst, img2, dst1, greenScreen);
	bitwise_or(dst, dst1, dst1);

	imshow("My Photo", img);
	imshow("Background image", img2);
	imshow("ũ�θ�Ű �ռ� ���", dst1);

	waitKey(0);
	return 0;
}