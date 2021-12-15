#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("image.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cout << "������ ���� �� ����" << endl;
		return (-1);
	}
	imshow("src", src);

	Mat erosion_dst, dilation_dst;
	Mat opening_dst, closing_dst;
	Mat gradient_dst;
	Mat skeleton_dst(src.size(), CV_8UC1, Scalar(0));

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(-1, -1));

	// ó�� �� �� ���·� skeletonization
	Mat src_clone = src.clone();
	Mat skeleton(src.size(), CV_8UC1, Scalar(0));
	threshold(src_clone, src_clone, 127, 255, THRESH_BINARY);
	Mat temp_1, eroded_1;
	Mat skel = src_clone.clone();
	do {
		erode(skel, eroded_1, kernel);
		dilate(eroded_1, temp_1, kernel);
		subtract(skel, temp_1, temp_1);
		bitwise_or(skeleton, temp_1, skeleton);
		eroded_1.copyTo(skel);
	} while ((countNonZero(skel) != 0));

	imshow("ó�� ���� ���ȭ", skeleton);


	// Gradient�� ������ ���� �� ��ȭ
	Mat src_edge;
	morphologyEx(src, src_edge, MORPH_GRADIENT, element2);
	int i = 0;
	do {
		src_edge = src_edge + src_edge;
		i++;
	} while (i == 255);

	// �������� ������� ��ü Ȯ��
	dilate(src_edge, dilation_dst, element);
	morphologyEx(dilation_dst, closing_dst, MORPH_CLOSE, element2);

	morphologyEx(closing_dst, closing_dst, MORPH_CLOSE, element2);
	Mat sum = dilation_dst + closing_dst;     // ������ ���� ��ü�� �� �巯���� �Ѵ�.
	morphologyEx(sum, closing_dst, MORPH_CLOSE, element2);
	morphologyEx(closing_dst, opening_dst, MORPH_OPEN, element2);

	Mat dst;
	sum = sum + opening_dst;
	morphologyEx(sum, sum, MORPH_CLOSE, element2);
	threshold(sum, dst, 127, 255, THRESH_OTSU);

	// �������� �� �κ�, ����Ǿ��ִ� �κ��� ���ش�.
	erode(dst, erosion_dst, element2);
	morphologyEx(erosion_dst, opening_dst, MORPH_OPEN, element2);

	// �и��� �������� skeletonization
	Mat temp, eroded;
	Mat skel_img = opening_dst.clone();
	do {
		erode(skel_img, eroded, kernel);
		dilate(eroded, temp, kernel);
		subtract(skel_img, temp, temp);
		bitwise_or(skeleton_dst, temp, skeleton_dst);
		eroded.copyTo(skel_img);
	} while ((countNonZero(skel_img) != 0));

	imshow("threshold", skeleton_dst);

	waitKey(0);
	return 0;
}