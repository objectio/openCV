#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int global_threshold_value = 130;
int threshold_type = 0;
const int max_value = 255;
const int max_binary_value = 255;
Mat src, src_gray, global_dst;
Mat otsu_dst, otsu_blur_dst, labels;
Mat centroids, img_color, stats;

static void MyThreshold(int, void*)
{
	//imshow("result", dst);
}

int main()
{
	Mat blur;
	int w500 = 0, w100 = 0;
	int w50 = 0, w10 = 0;

	src = imread("coin.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cout << "������ ���� �� ����" << endl;
		return (-1);
	}
	imshow("result", src);

	threshold(src, global_dst, global_threshold_value, max_binary_value, THRESH_BINARY); // global Thresholding
	imshow("Global Thresholding", global_dst);

	threshold(src, otsu_dst, 0, max_binary_value, THRESH_OTSU); // otsu Thresholding
	imshow("Otsu", otsu_dst);

	Size size = Size(15, 15);
	GaussianBlur(src, blur, size, 0);
	threshold(blur, otsu_blur_dst, 0, max_binary_value, THRESH_OTSU);
	imshow("Otsu after Blurring", otsu_blur_dst);

	int n = connectedComponentsWithStats(otsu_blur_dst, labels, stats, centroids);

	vector<Vec3b> colors(n + 1);
	colors[0] = Vec3b(0, 0, 0);
	for (int i = 1; i <= n; i++) {
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}
	img_color = cv::Mat::zeros(src.size(), CV_8UC3);
	for (int y = 0; y < img_color.rows; y++)
		for (int x = 0; x < img_color.cols; x++)
		{
			int label = labels.at<int>(y, x);
			img_color.at<Vec3b>(y, x) = colors[label];
		}
	imshow("Labeled map", img_color);
	for (int i = 0; i < n; i++)
	{
		cout << i << "��° ���� : ";
		int area = stats.at<int>(i, CC_STAT_AREA);
		cout << area << " ";
		int width = stats.at<int>(i, CC_STAT_WIDTH);
		cout << width << " ";
		int height = stats.at<int>(i, CC_STAT_HEIGHT);
		cout << height << " " << endl;

		if (4970 <= area && area < 100000) w500++;
		else if (3980 <= area && area < 4970) w100++;
		else if (2900 <= area && area < 3980) w50++;
		else if (2100 <= area && area < 2900) w10++;
	}
	cout << "\n�� " << 500 * w500 + 100 * w100 + 50 * w50 + 10 * w10 << "���Դϴ�!\n";
	cout << "5��� " << w500 << "��\n";
	cout << "1��� " << w100 << "��\n";
	cout << "5�ʿ� " << w50 << "��\n";
	cout << "1�ʿ� " << w10 << "��\n\n";

	createTrackbar("�Ӱ谪", "result", &global_threshold_value, max_value, MyThreshold);
	MyThreshold(0, 0);

	waitKey();
	return 0;
}