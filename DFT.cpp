#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void log_mag(Mat complex_mat, Mat& dst)
{
	Mat planes[2];
	split(complex_mat, planes);
	magnitude(planes[0], planes[1], dst);
	log(dst + 1, dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8U);
}

void shuffling(Mat mag_img, Mat& dst)
{
	int cx = mag_img.cols / 2;
	int cy = mag_img.rows / 2;
	Rect q1(cx, 0, cx, cy);
	Rect q2(0, 0, cx, cy);
	Rect q3(0, cy, cx, cy);
	Rect q4(cx, cy, cx, cy);

	dst = Mat(mag_img.size(), mag_img.type());
	mag_img(q1).copyTo(dst(q3));
	mag_img(q3).copyTo(dst(q1));
	mag_img(q2).copyTo(dst(q4));
	mag_img(q4).copyTo(dst(q2));
}

Mat DFT_1D(Mat one_row, int dir)
{
	int N = (int)one_row.total();
	Mat dst(one_row.size(), CV_32FC2);
	
	for (int k = 0; k < N; k++) {
		Vec2f complex(0, 0);
		for (int n = 0; n < N; n++)
		{
			float theta = (float)(dir * (-2) * CV_PI * k * n / N);
			Vec2f value = one_row.at<Vec2f>(n);
			complex[0] += value[0] * cos(theta) - value[1] * sin(theta);
			complex[1] += value[1] * cos(theta) + value[0] * sin(theta);
		}
		dst.at<Vec2f>(k) = complex;
	}
	if (dir == -1) dst /= N;
	return dst;
}

void DFT_2D(Mat complex_img, Mat& dst, int dir)
{
	complex_img.convertTo(complex_img, CV_32F);
	Mat tmp(complex_img.size(), CV_32FC2, Vec2f(0, 0));
	tmp.copyTo(dst);

	for (int i = 0; i < complex_img.rows; i++) {
		Mat one_row = complex_img.row(i);
		Mat dft_row = DFT_1D(one_row, dir);
		dft_row.copyTo(tmp.row(i));
	}

	transpose(tmp, tmp);
	for (int i = 0; i < tmp.rows; i++) {
		Mat one_row = tmp.row(i);
		Mat dft_row = DFT_1D(tmp.row(i), dir);
		dft_row.copyTo(dst.row(i));
	}
	transpose(dst, dst);
}

void DFT(Mat image, Mat& dft_coef, Mat& shuffling_coef)
{
	Mat complex_img;
	Mat tmp[] = { image, Mat::zeros(image.size(), CV_8U) };

	merge(tmp, 2, complex_img);
	DFT_2D(complex_img, dft_coef, 1);
	shuffling(dft_coef, shuffling_coef);
}

void IDFT(Mat dft_coef, Mat& idft_img)
{
	Mat idft_coef;
	Mat t[2];

	shuffling(dft_coef, dft_coef);
	DFT_2D(dft_coef, idft_coef, -1);
	split(idft_coef, t);
	t[0].convertTo(idft_img, CV_8U);
}

Mat getLowpassFilter(Size size, int radius)
{
	Point center = size / 2;
	Mat filter(size, CV_32FC2, Vec2f(0, 0));
	circle(filter, center, radius, Vec2f(1, 1), -1);
	return filter;
}

Mat getHighpassFilter(Size size, int radius)
{
	Point center = size / 2;
	Mat filter(size, CV_32FC2, Vec2f(1, 1));
	circle(filter, center, radius, Vec2f(0, 0), -1);
	return filter;
}

int main()
{
	Mat src, dft_coef, shuffling_coef;
	Mat dft_img, shuffling_img, idft_img;
	Mat low_mat, high_mat, low_dft_img;
	Mat high_dft_img, low_idft_img, high_idft_img;

	src = imread("image.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cout << "영상을 읽을 수 없음" << endl;
		return (-1);
	}
	imshow("Source", src);

	DFT(src, dft_coef, shuffling_coef);
	log_mag(dft_coef, dft_img);
	log_mag(shuffling_coef, shuffling_img);

	int radius = 50;
	Mat lf = getLowpassFilter(dft_coef.size(), radius);
	radius = 10;
	Mat hf = getHighpassFilter(dft_coef.size(), radius);

	multiply(shuffling_coef, lf, low_mat);
	multiply(shuffling_coef, hf, high_mat);
	log_mag(low_mat, low_dft_img);
	log_mag(high_mat, high_dft_img);

	IDFT(low_mat, low_idft_img);
	IDFT(high_mat, high_idft_img);

	imshow("Low DFT", low_dft_img);
	imshow("High DFT", high_dft_img);
	imshow("SHUFFLING", shuffling_img);
	imshow("Low Inverse DFT", low_idft_img);
	imshow("High Inverse DFT", high_idft_img);

	waitKey(0);
	return 0;
}