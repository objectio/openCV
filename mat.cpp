#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int red, green, blue;
int drawing = false;

void on_trackbar(int, void*) {}

void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat& img = *(Mat*)(param);
	if (event == EVENT_LBUTTONDOWN) {
		drawing = true;
	}
	else if (event == EVENT_MOUSEMOVE) {
		if (drawing == true)
			circle(img, Point(x, y), 3, Scalar(blue, green, red), 10);
	}
	else if (event == EVENT_LBUTTONUP)
		drawing = false;
	imshow("src", img);
}

void brighten(Mat& img)
{
	int value = 0;
	cout << "밝기 값을 입력하시오: "; cin >> value;
	for (int r = 0; r < img.rows; r++)
		for (int c = 0; c < img.cols; ++c)
			for (int ch = 0; ch < 3; ++ch)
			img.at<Vec3b>(r, c)[ch] = saturate_cast<uchar>(img.at<Vec3b>(r, c)[ch] + value);
}

int contrastEnh(int input, int x1, int y1, int x2, int y2)
{
	double output;
	if (0 <= input && input <= x1) {
		output = y1 / x1 * input;   // 기울기 y1 / x1 
	}
	else if (x1 < input && input <= x2) {
		output = ((y2 - y1) / (x2 - x1)) * (input - x1) + y1;
	}
	else if (x2 < input && input <= 255) {
		output = ((255 - y2) / (255 - x2)) * (input - x2) + y2;
	}
	return (int)output;
}

void contrast(Mat& img)
{
	int x1, y1, x2, y2;
	cout << "x1 값을 입력하시오: "; cin >> x1;
	cout << "y1 값을 입력하시오: "; cin >> y1;
	cout << "x2 값을 입력하시오: "; cin >> x2;
	cout << "y2 값을 입력하시오: "; cin >> y2;

	for (int r = 0; r < img.rows; r++) {
		for (int c = 0; c < img.cols; c++) {
			for (int ch = 0; ch < 3; ch++) {
				int output = contrastEnh(img.at<Vec3b>(r, c)[ch], x1, y1, x2, y2);
				img.at<Vec3b>(r, c)[ch] = saturate_cast<uchar>(output);
			}
		}
	}
	imshow("CONTRAST", img);
}

void invert(Mat& img)
{
	for (int r = 0; r < img.rows; r++) {
		for (int c = 0; c < img.cols; c++) {
			for (int ch = 0; ch < 3; ch++) {
				img.at<Vec3b>(r, c)[ch] = 255 - img.at<Vec3b>(r, c)[ch];
			}
		}
	}
	imshow("INVERT", img);
}

void binary(Mat& img)
{
	Mat gray(img.rows, img.cols, CV_8UC1);
	Mat bw(img.rows, img.cols, CV_8UC1);

	for (int h = 0; h < img.rows; h++) {
		for (int w = 0; w < img.cols; w++) {
			gray.at<uchar>(h, w) = (img.at<Vec3b>(h, w)[0] + img.at<Vec3b>(h, w)[1] + img.at<Vec3b>(h, w)[2]) / 3;

			if (gray.at<uchar>(h, w) < 128) {
				bw.at<uchar>(h, w) = 0;
			}
			else {
				bw.at<uchar>(h, w) = 255;
			}
		}
	}
	imshow("BINARIZATION", bw);
}

void gamma_correction(Mat& img) {
	Mat dst;
	double gamma = 0.5;
	cout << "감마 값을 입력하시오: "; cin >> gamma;
	Mat table(1, 256, CV_8U);
	uchar* p = table.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast <uchar> (pow(i / 255.0, gamma) * 255.0);
	LUT(img, table, dst);
	imshow("GAMMA CORRECTION", dst);
}

int main()
{
	Mat image = imread("myface.jpg");
	Mat img = image.clone();
	double gamma = 0.5;
	
	namedWindow("src", 1);
	imshow("src", img);
	imshow("My Photo", image);
	setMouseCallback("src", onMouse, &img);

	createTrackbar("R", "src", &red, 255, on_trackbar);
	createTrackbar("G", "src", &green, 255, on_trackbar);
	createTrackbar("B", "src", &blue, 255, on_trackbar);
	
	while (1)
	{
		int key = waitKeyEx();
		cout << key << " ";

		if (key == 'q') break;
		else if (key == '1') {
			Mat oimage = image.clone();
			brighten(oimage);
			imshow("BRIGHT", oimage);
		}
		else if (key == '2') {
			Mat oimage = image.clone();
			contrast(oimage);
		}
		else if (key == '3') {
			Mat oimage = image.clone();
			invert(oimage);
		}
		else if (key == '4') {
			Mat oimage = image.clone();
			binary(oimage);
		}
		else if (key == '5') {
			Mat oimage = image.clone();
			gamma_correction(oimage);
		}
		else if (key == '6') {
			Mat oimage = image.clone();
			bitwise_xor(image, img, oimage);
			imshow("#얼스타그램", oimage);
		}
	}
	return 0;
}
