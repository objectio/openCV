#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

Mat img;
Mat train_features(900, 784, CV_32FC1);
Mat labels(900, 1, CV_32FC1);
Ptr<ml::KNearest> knn;
Ptr<ml::TrainData> trainData;

void test1()
{
	// 1번째 테스트 단계
	cout << "---------------------1번째 테스트---------------------" << '\n';

	int right = 0;
	float right_rate = 0;

	Mat predictedLabels;
	for (int i = 0; i < 900; i++) {
		Mat test = train_features.row(i);
		knn->findNearest(test, 3, predictedLabels);      // k = 3
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 샘플" << i << "의 라벨 = " << prediction << '\n';
		if (i / 90 == prediction) right++;
	}
	cout << "갯수: " << right << '\n';
	right_rate = (double)right / 900 * 100;
	cout << "정답률 : " << right_rate << '\n\n';
}

void test2()
{

}

void test3()
{
	// 3번째 테스트 단계
	cout << "---------------------3번째 테스트---------------------" << '\n';
	Mat img = imread("4-train.png", IMREAD_GRAYSCALE);
	// 4-train.png 이미지 골격화
	threshold(img, img, 127, 255, THRESH_BINARY);

	imshow("a", img);
	Mat skel(img.size(), CV_8UC1, Scalar(0));  // skeleton = 0
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
	Mat temp, eroded;
	
	do {
		erode(img, eroded, element);
		dilate(eroded, temp, element);
		subtract(img, temp, temp);
		bitwise_or(skel, temp, skel);
		eroded.copyTo(img);
	} while ((countNonZero(img) != 0));
	
	imshow("re", skel);
	
	// 각 의상 영상을 row vector로 만들어 train_features에 저장
	for (int r = 0; r < 30; r++) {                       // 세로로 30개
		for (int c = 0; c < 30; c++) {                   // 가로로 30개
			int i = 0;
			for (int y = 0; y < 28; y++) {               // 28 X 28
				for (int x = 0; x < 28; x++) {
					train_features.at<float>(r * 30 + c, i++) = skel.at<uchar>(r * 28 + y, c * 28 + x);
				}
			}
		}
	}

	// 각 의상 영상에 대한 레이블을 저장한다.
	for (int i = 0; i < 90; i++)  // top
		labels.at<float>(i, 0) = 0;
	for (int i = 180; i < 270; i++) 
		labels.at<float>(i, 0) = 0;
	for (int i = 360; i < 450; i++)
		labels.at<float>(i, 0) = 0;
	for (int i = 540; i < 630; i++)
		labels.at<float>(i, 0) = 0;

	for (int i = 90; i < 180; i++)  // pants
		labels.at<float>(i, 0) = 1;
	for (int i = 270; i < 360; i++)  // dress
		labels.at<float>(i, 0) = 2;
	for (int i = 450; i < 540; i++)  // shoes
		labels.at<float>(i, 0) = 3;
	for (int i = 630; i < 720; i++)  // sneakers
		labels.at<float>(i, 0) = 4;
	for (int i = 720; i < 810; i++)  // bag
		labels.at<float>(i, 0) = 5;
	for (int i = 810; i < 900; i++)  // boots
		labels.at<float>(i, 0) = 6;

	// 학습 단계
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, labels);
	knn->train(trainData);

	int right = 0;
	float right_rate2;
	Mat predictedLabels;
	for (int i = 0; i < 900; i++) {
		Mat test = train_features.row(i);
		knn->findNearest(test, 3, predictedLabels);      // k = 3
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 샘플" << i << "의 라벨 = " << prediction << '\n';
		if (i < 90 && prediction == 0) right++;
		else if (i < 180 && prediction == 1) right++;
		else if (i < 270 && prediction == 0) right++;
		else if (i < 360 && prediction == 2) right++;
		else if (i < 450 && prediction == 0) right++;
		else if (i < 540 && prediction == 3) right++;
		else if (i < 630 && prediction == 0) right++;
		else if (i < 720 && prediction == 4) right++;
		else if (i < 810 && prediction == 5) right++;
		else if (i < 900 && prediction == 6) right++;
	}
	cout << "갯수: " << right << '\n';
	right_rate2 = ((double)right / 900);
	double rate = right_rate2;
	cout << "정답률 : " << rate * 10 << '\n\n';
}

int main()
{
	// 4-train.png 이미지 로드 - 한 데이터 당 크기가 28 X 28이므로 양선형 보간법을 이용해 30 X 30으로 늘린다.
	Mat img = imread("4-train.png", IMREAD_GRAYSCALE);

	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", img);

	// 각 의상 영상을 row vector로 만들어 train_features에 저장
	for (int r = 0; r < 30; r++) {                       // 세로로 30개
		for (int c = 0; c < 30; c++) {                   // 가로로 30개
			int i = 0;
			for (int y = 0; y < 28; y++) {               // 28 X 28
				for (int x = 0; x < 28; x++) {
					train_features.at<float>(r * 30 + c, i++) = img.at<uchar>(r * 28 + y, c * 28 + x);
				}
			}
		}
	}

	// 각 의상 영상에 대한 레이블을 저장한다.
	for (int i = 0; i < 900; i++) {
		labels.at<float>(i, 0) = (i / 90);      // 의상종류별로 의상데이터 90개씩
	}

	// 학습 단계
	knn = ml::KNearest::create();
	trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, labels);
	knn->train(trainData);


	
	// 2번째 테스트
	int p = 29;
	Mat testData = imread("4-test.jfif", IMREAD_GRAYSCALE);
	//rectangle(testData, Rect(28 * p + (2 * p + 1), 28 * p + (2 * p + 1), 28, 28), Scalar(255, 255, 255), 1);
	//rectangle(testData, Rect(30 * p, 30 * p, 30, 30), Scalar(255, 255, 255), 1);
	imshow("testt", testData);

	int n = 0, m = 8;
	
	Mat roi[20];
	
	for (int i = 0; i < 10; i++) {
		roi[2 * i + 1] = testData(Rect(28 * (3 * i + n + 1) + (2 * (3 * i + n + 1) + 1), 28 * m + (2 * m + 1), 28, 28));
	}
	for (int i = 0; i < 10; i++) {
		roi[2 * i] = testData(Rect(28 * (3 * i + n) + (2 * (3 * i + n) + 1), 28 * m + (2 * m + 1), 28, 28));
	}
	//imshow("testt", roi[3]);
	// test#.jpg 파일로 저장. 4-test.jfif 이미지에서는 T-shirt, pants, dress, jacket, shoes, shirt, blouse, sneakers, bag, boots 순이므로 Test Data 순서에 맞춰 저장한다.
	//imwrite("test0.jpg", roi[0]);	imwrite("test1.jpg", roi[1]);	imwrite("test2.jpg", roi[2]);	imwrite("test3.jpg", roi[3]);
	//imwrite("test4.jpg", roi[10]);	imwrite("test5.jpg", roi[11]);	imwrite("test6.jpg", roi[4]);	imwrite("test7.jpg", roi[5]);
	//imwrite("test8.jpg", roi[6]);	imwrite("test9.jpg", roi[7]);	imwrite("test10.jpg", roi[8]);	imwrite("test11.jpg", roi[9]);
	//imwrite("test12.jpg", roi[12]);	imwrite("test13.jpg", roi[13]);	imwrite("test14.jpg", roi[14]);	imwrite("test15.jpg", roi[15]);
	//imwrite("test16.jpg", roi[16]);	imwrite("test17.jpg", roi[17]);	imwrite("test18.jpg", roi[18]);	imwrite("test19.jpg", roi[19]);
	
	
	roi[0] = imread("test0.jpg", 0);		roi[1] = imread("test1.jpg", 0);		roi[2] = imread("test2.jpg", 0);		roi[3] = imread("test3.jpg", 0);
	roi[4] = imread("test4.jpg", 0);		roi[5] = imread("test5.jpg", 0);		roi[6] = imread("test6.jpg", 0);		roi[7] = imread("test7.jpg", 0);
	roi[8] = imread("test8.jpg", 0);		roi[9] = imread("test9.jpg", 0);		roi[10] = imread("test10.jpg", 0);		roi[11] = imread("test11.jpg", 0);
	roi[12] = imread("test12.jpg", 0);		roi[13] = imread("test13.jpg", 0);		roi[14] = imread("test14.jpg", 0);		roi[15] = imread("test15.jpg", 0);
	roi[16] = imread("test16.jpg", 0);		roi[17] = imread("test17.jpg", 0);		roi[18] = imread("test18.jpg", 0);		roi[19] = imread("test19.jpg", 0);

	// test#.jpg 이미지 모아서 한 윈도우에 출력
	Mat test2(roi[0].cols * 2, roi[0].rows * 10, CV_8U);
	Mat dst_roi[20];

	int k = 0;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 2; j++) {
			dst_roi[k] = test2(Rect(roi[k].cols * i, roi[k].rows * j, roi[k].cols, roi[k].rows));
			roi[k].copyTo(dst_roi[k]);
			k++;
		}
	}
	
	imshow("test2", test2);

	// test#.jpg의 각 의상 영상을 row vector로 만들어 test_features에 저장
	Mat test_features(20, 784, CV_32FC1);

	for (int r = 0; r < 2; r++) {                       // 세로로 2개
		for (int c = 0; c < 10; c++) {                   // 가로로 10개
			int i = 0;
			for (int y = 0; y < 28; y++) {               // 30 X 30
				for (int x = 0; x < 28; x++) {
					test_features.at<float>(r * 10 + c, i++) = test2.at<uchar>(r * 28 + y, c * 28 + x);
				}
			}
		}
	}
	
	int right = 0;
	int right_rate = 0;

	Mat predictedLabels;
	for (int i = 0; i < 20; i++) {
		Mat test = test_features.row(i);
		knn->findNearest(test, 3, predictedLabels);      // k = 3
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 샘플" << i << "의 라벨 = " << prediction << '\n';
		if (i / 2 == prediction) right++;
	}
	cout << "갯수: " << right << '\n';
	right_rate = (double)right / 20;
	cout << "정답률 : " << right_rate << '\n\n';
	
	while (1)
	{
		int key = waitKeyEx();
		cout << key << " ";

		if (key == 'q') break;
		else if (key == '1') {
			test1();
		}
		else if (key == '2') {
			
		}
		else if (key == '3') {
			test3();
		}
	}

	waitKey();
	return (0);
}