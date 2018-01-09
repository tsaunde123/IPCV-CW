/*
COMS30121 - thr.cpp
TOPIC: RGB explicit thresholding
Getting-Started-File for OpenCV
University of Bristol
Finish first part and question 1 for second part.
2017-11-23 16:39:58
rewrite the magnitude function and the normalization function.
2017-11-24 12:08:28
image size:640*426
2017-11-24 21:42:55
finished the line detection function and it works for most of the case(images).
fix the problem in printing
line function.
2017-11-24 23:59:05
Test the image using build in function. (test the old version)
2017-11-25 02:03:28
start with function of the rectangles.
2017-11-26 00:27:35
fail to test with 0.5 degree interval. Line 219,223,224,231 need to change when test interval.
0.5 interval test too many lines, 2 degree 3 lines for 1 actually line, 3 degree 2 for 1.
2017-11-28 16:06:07
Test with small box.
2017-12-1 15:40:45
Rewrite the covelution function to get better result in deirection matrix.
2017-12-1 17:02:40
Hough circle works for most of images, and remove the overlap circles.
Next redo the valid box.
2017-12-2 20:41:53
check the valid boxes function works, and remove the overlap rectangles above
a detected circle, and pick the best one.
2017-12-3 02:10:40
add line function for the detection.
2017-12-3 15:22:35
Final version of part 3.
*/


#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <opencv2/opencv.hpp>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "time.h"
#include "functions.h"
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#define pi 3.1415926

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

Mat kernel_x = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
Mat kernel_y = (Mat_<int>(3, 3) << -1,-2,-1,  0, 0, 0,  1, 2, 1);

/** Global variables */
//String cascade_name = "frontalface.xml";
 String cascade_name = "cascade.xml";
CascadeClassifier cascade;

bool sortRects(const cv::Rect &r1, const cv::Rect &r2);
/** @function detectAndDisplay */
void detectAndDisplay(Mat frame, vector<Rect> &rects){
	std::vector<Rect> faces;
	Mat frame_gray;  //As the frame is already gray image.
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
	sort(faces.begin(), faces.end(), sortRects);
	//// 3. Print number of Faces found
	//for (int i = 0; i < faces.size(); i++)
	//{
	//rectangle(frame, Point(faces[i].x, faces[i].y),
	//  	Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2);
	//}

	rects = faces;
	// cout << "Dart num : " << faces.size() << endl;
}

void normalization(cv::Mat &gray_image){ // at here, the gray image is type of int, not char.
	normalize(gray_image, gray_image, 0, 255, 32, -1);
}

void convolution(cv::Mat &input, cv::Mat &kernel, cv::Mat &output){
	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE);
	// now we can do the convoltion
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			double sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
			{
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					double  imageval = (double)paddedInput.at<uchar>(imagex, imagey);
					int kernalval = kernel.at<int>(kernelx, kernely);

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			output.at<double>(i, j) = (double)sum;
		}
	}
}

void direction(cv::Mat &gray_image1, cv::Mat &gray_image2, cv::Mat &output) { // image1 is dx.
	int min = 0;
	int max = 0;
	for (int y = 0; y < gray_image1.rows; y++) {
		for (int x = 0; x < gray_image1.cols; x++) {
			int pixeldx = gray_image1.at<double>(y, x);
			int pixeldy = gray_image2.at<double>(y, x);
			float direct = (atan2((float)pixeldy, (float)pixeldx));
			output.at<float>(y, x) = direct;
			// printf("direction = %f\n", direct);
		}
	}
}

void magnitude(cv::Mat &gray_image1, cv::Mat &gray_image2, cv::Mat &output) { // image1 is dx.
	for (int y = 0; y < gray_image1.rows; y++) {
		for (int x = 0; x < gray_image1.cols; x++) {
			int pixeldx = gray_image1.at<double>(y, x);
			int pixeldy = gray_image2.at<double>(y, x);
			int magni = cvRound(sqrt(pow(pixeldx, 2) + pow(pixeldy, 2)));
			output.at<int>(y, x) = magni;
		}
	}
	// normalization(output);
}

void thresholds(cv::Mat &gray_image, cv::Mat &output) {
	for (int y = 0; y < gray_image.rows; y++) {
		for (int x = 0; x < gray_image.cols; x++) {
			if (gray_image.at<int>(y, x) < 100) {
				output.at<uchar>(y, x) = 0;
			}
			else {
				output.at<uchar>(y, x) = 255;
			}
		}
	}
}

void searchLine(cv::Mat &gray_image, cv::Mat &output, vector<Vec4i> &lines){

	/*vector<Vec4i> lines;
	outputlines = lines;*/
    HoughLinesP( gray_image, lines, 1, 2*CV_PI/180, 30, 50, 10 );
    // cout << "Line number real: " << lines.size() << std::endl;
    for( size_t i = 0; i < lines.size(); i++){
        line( output, Point(lines[i][0], lines[i][1]),
            Point(lines[i][2], lines[i][3]), cvScalar(255), 1, 8, 0);
    }

}

bool checkIfParallelLine(vector<Vec4i> &lines) {
	vector<Vec2i> degreeTable;
	for (int i = 0; i < lines.size(); i++) {
		Point p1(lines[i][0], lines[i][1]);
		Point p2(lines[i][2], lines[i][3]);
		int angle = cvRound(atan2(p1.y - p2.y, p1.x - p2.x) * (180 / pi));

		bool newElement = true;
		for (int j=0; j < degreeTable.size(); j++) {
			if (degreeTable[j][0]<(angle+5) && degreeTable[j][0]>(angle-5)) {
				degreeTable[j][1]++;
				newElement = newElement && false;
			}
		}
		if (newElement) {
			degreeTable.push_back(cv::Vec2i(angle, 1));
		}
	}

	bool Notparallel = true;
	for (int j=0; j < degreeTable.size(); j++) {
		if (degreeTable[j][1]>6) {
			Notparallel = Notparallel && false;
			return true;
		}
	}
	if (Notparallel) {
		return false;
	}
}

bool sortRects(const cv::Rect &r1, const cv::Rect &r2) {
	return r1.area() > r2.area();
}

void removeOverlaplineRect(vector<Rect> &lineRects, vector<Rect> &outputRects){
	if (lineRects.size()==1) {
		outputRects = lineRects;
	}
	else if (lineRects.size() > 1) {
		sort(lineRects.begin(), lineRects.end(), sortRects);
		outputRects.push_back(lineRects[0]);
	}

}

void removeFalsePositive(Mat img_object, Mat image,vector<Rect> &allValidRects) {
	Rect r = allValidRects[0];
	Mat validRect = image(r);
	//sudf(img_object, validRect);
}

void validBoxes(Mat frame, vector<cv::Point> &circleCenters, vector<Rect> &rects, vector<int> lineGroup, vector<Rect> &allValideRects){

	int bestFitRect = 0;
	int Centers[30] = {0};
	vector<Rect> LineRects;
	//vector<Rect> allValideRects;

	for (int k = 0; k < circleCenters.size(); k++) {
		//vector<cv::Point> relateRectsCenter;
		std::vector<Vec3f> relateRectsCenter;
		for (int i = 0; i < rects.size(); i++) {
			Centers[i] = 0;
			if (circleCenters[k].x >= rects[i].x && circleCenters[k].x <= (rects[i].x + rects[i].width) && circleCenters[k].y >= rects[i].y && circleCenters[k].y <= (rects[i].y + rects[i].height)) {
				relateRectsCenter.push_back(cv::Vec3f(rectangleCenter(rects[i]).x, rectangleCenter(rects[i]).y, i));
				Centers[i]++;
			}
		}
		bestFitRect = removeOverlapRect(relateRectsCenter, circleCenters[k]);
		Centers[bestFitRect]++;
	}
	//cout << bestFitRect << std::endl;
	//cout << Centers[bestFitRect] << std::endl;

	vector<int> valid_index;
	int houghlineBox = 0;
	for( int i = 0; i < rects.size(); i++ ){
		// ------ Center>1 means best fit rect, ==1 means overlaped one(removed), ==0 means no circle in side.
		if(Centers[i] > 1){

			allValideRects.push_back(rects[i]);
			rectangle(frame, Point(rects[i].x, rects[i].y), Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), Scalar( 0, 255, 0 ), 2);
		}
		else if (Centers[i] < 1) {
			houghlineBox++;
			if (lineGroup[i]>10) {
				if (!isIntersects(rects[bestFitRect], rects[i])) {
					LineRects.push_back(rects[i]);
				}
			}
		}
	}
	cout << "Number of box process line searching: " << houghlineBox << endl;


	vector<Rect> validLineRects;
	removeOverlaplineRect(LineRects, validLineRects);
	for (int i = 0; i < validLineRects.size(); i++) {
		allValideRects.push_back(validLineRects[i]);
		rectangle(frame, Point(validLineRects[i].x, validLineRects[i].y),
			Point(validLineRects[i].x + validLineRects[i].width, validLineRects[i].y + validLineRects[i].height), Scalar(255, 255, 0), 2);
	}
	cout << "Valid box number after searching line feature: " << validLineRects.size() << endl;
}

int main(int argc, const char** argv) {

	//construct a window for image display
	namedWindow("Display window1", CV_WINDOW_AUTOSIZE);
	Mat gray_image;
	Mat image;
	Mat img_object;

	img_object = imread("dart.bmp");

	//for (int imageIndex=0; imageIndex<16; imageIndex++){ // test for 1 image.


		// ------------------------ viol and jones method -----------------------------
		//string imageName = "dart" + to_string(imageIndex) + ".jpg";
		// ---------------------- for image 8, box 4, 5, 10, 11 should detect lines.
		//image = imread("Images/" + imageName , 1);
		image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
		int imageIndex = 0;
		int slen = strlen(argv[1]);
		if (slen == 9) {
			imageIndex = (int)argv[1][4] - '0';
		}
		else if (slen == 10) {
			int num1 = (int)argv[1][4] - '0';
			int num2 = (int)argv[1][5] - '0';
			imageIndex = num1 * 10 + num2;
		}

		std::cout << "------------------------------- Image " + to_string(imageIndex) + " -------------------------------" << std::endl;


		 // 2. Load the Strong Classifier in a structure called `Cascade'
		 if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };
		 // 3. Detect Faces and Display Result
		 std::vector<Rect> rectsMulti;
		 detectAndDisplay(image, rectsMulti);
		 Mat edges = Mat::zeros(gray_image.rows, gray_image.cols, DataType<uchar>::type);
		 Canny(image, edges, 100, 200, 3, true);

		 // -------------------------edge detection for the images----------------------------------
		 cv::cvtColor(image, gray_image, CV_BGR2GRAY);
		 Mat tmp_dx = Mat::zeros(gray_image.rows, gray_image.cols, DataType<double>::type);
		 Mat tmp_dy = Mat::zeros(gray_image.rows, gray_image.cols, DataType<double>::type);
		 convolution(gray_image, kernel_x, tmp_dx);  //0 is the number for the image.
		 convolution(gray_image, kernel_y, tmp_dy);

		 Mat tmp_direction = Mat::zeros(gray_image.rows, gray_image.cols, DataType<float>::type);
		 direction(tmp_dx, tmp_dy, tmp_direction);

		 Mat tmp_magnitude = Mat::zeros(gray_image.rows, gray_image.cols, DataType<int>::type);
		 magnitude(tmp_dx, tmp_dy, tmp_magnitude);

		 Mat tmp_thresholds = Mat::zeros(gray_image.rows, gray_image.cols, DataType<uchar>::type);
		 thresholds(tmp_magnitude, tmp_thresholds);



		// //-------------------- searching circles in the images ------------------------------
		 vector<cv::Point> circleCenters;
		 std::vector<Vec3f> circles;
		 std::vector<Vec3f> circlesNoOverlap;
		 houghCircles(edges, edges, tmp_direction, circles, circleCenters);
		 // ----------- storing circle -------------
		 sort(circles.begin(),circles.end(),sortCircles);

		 // -------- for presenting
		 cout << "---------- " << "Hough circle processing " << "----------" << endl;
		 cout << "Total circle number detected: " << circles.size() << endl;
		 for (int i=0; i<circles.size(); i++){
		 	cout << "circle radius: " << circles[i][2] << endl;
		 }
		 // -------- end

		 removeOverlapCircle(circles, circlesNoOverlap);
		 cout << "Circle number after removing overlaped: " << circlesNoOverlap.size() << endl;
		 for (int i=0; i<circlesNoOverlap.size(); i++){

		 	Point center(cvRound(circlesNoOverlap[i][0]), cvRound(circlesNoOverlap[i][1]));

		 	int radius = (circlesNoOverlap[i][2]);
		 	circle(edges, center, radius, cvScalar(255), 3, 8, 0);
		 	cout << "Remained circle radius: " << circlesNoOverlap[i][2] << endl;

		 }

		 // --------------- searching line in all boxes(i is the index of box) ----------------

		 cout << "---------- " << "Hough line processing " << "----------" << endl;
		 vector<int> lineGroup;
		 vector<Vec4i> lines;
		 int lineNum = 0;
		 for (int i = 0; i<rectsMulti.size(); i++) {
			 Rect r = rectsMulti[i];
			 Mat Box = edges(r);
			 Mat line_result = Mat::zeros(Box.rows, Box.cols, DataType<int>::type);

			 searchLine(Box, line_result, lines);
			 bool parallel = checkIfParallelLine(lines);
			 if (parallel) {
				 lineGroup.push_back(0);
			 }
			 else {
				 lineGroup.push_back(lines.size());
			 }
		 }

		 // ---------------------- Draw box around faces found and find only the valid boxes -----------------------
		  for (int i = 0; i < rectsMulti.size(); i++)
		  {
		  	rectangle(edges, Point(rectsMulti[i].x, rectsMulti[i].y),
		  		Point(rectsMulti[i].x + rectsMulti[i].width, rectsMulti[i].y + rectsMulti[i].height), cvScalar(255), 2);
		  }

		 // ----------------------- remove boxes without circle inside, we also need to use line later ------
		 vector<Rect> allValideRects;
		 validBoxes(image, circleCenters, rectsMulti, lineGroup, allValideRects);

		 // -------------------------- calculate the ground truth --------------------------------------------
		 cout << "---------- F1 calculation ----------" << endl;
		 vector<vector<Rect>> groundtruth;
		 groundTruth(groundtruth);
		 float f1 = f1_score(allValideRects, image, imageIndex, groundtruth);

		 //--------------------visualise the loaded image in the window--------------------------------------------
		imwrite("image" + to_string(imageIndex) + ".jpg",image);
		imwrite("gray_image.jpg",gray_image);
		imwrite("dx" + to_string(imageIndex) + ".jpg", tmp_dx);
		imwrite("dy" + to_string(imageIndex) + ".jpg", tmp_dy);
		imwrite("direc" + to_string(imageIndex) + ".jpg", tmp_direction);
		imwrite("magni" + to_string(imageIndex) + ".jpg", tmp_magnitude);
		imwrite("thres" + to_string(imageIndex) + ".jpg", tmp_thresholds);
		imwrite("edges" + to_string(imageIndex) + ".jpg", edges);

	//}

	cv::imshow("Display window1", image);
	cv::waitKey(0);
	return 0;

}
