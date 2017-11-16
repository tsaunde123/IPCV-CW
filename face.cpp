/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#define pi 3.1415926

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat &frame, Mat &output );
void detectCircles(Mat &thres_frame, Mat &output);
void sobelConvolution(cv::Mat &gray_image, cv::Mat &output, int ker[], int image_num);
void direction(cv::Mat &gray_image1, cv::Mat &gray_image2, cv::Mat &output);
void magnitude(cv::Mat &gray_image1, cv::Mat &gray_image2, cv::Mat &output);
void thresholds(cv::Mat &gray_image, cv::Mat &output);

int kernelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
int kernelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
int max_val[3] = {90,   0, 0};
int min_val[3] = {-100, 0, 0};
int max_mag[3] = { 294,   0, 0 };
int min_mag[3] = { 0  , 0, 0 };

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat frame_output = frame;
	Mat gray_frame;
	cvtColor(frame, gray_frame, CV_BGR2GRAY);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };


	//dlur image (convolution).
	Mat tmp_image1 = Mat::zeros(gray_frame.rows, gray_frame.cols, DataType<int>::type);
	Mat tmp_image2 = Mat::zeros(gray_frame.rows, gray_frame.cols, DataType<int>::type);
	sobelConvolution(gray_frame, tmp_image1, kernelX, 0);  //0 is the number for the image.
	sobelConvolution(gray_frame, tmp_image2, kernelY, 0);

	Mat tmp_image3 = Mat::zeros(gray_frame.rows, gray_frame.cols, DataType<float>::type);
	direction(tmp_image1, tmp_image2, tmp_image3);

	Mat tmp_image4 = Mat::zeros(gray_frame.rows, gray_frame.cols, DataType<int>::type);
	magnitude(tmp_image1, tmp_image2, tmp_image4);

	Mat tmp_image5 = Mat::zeros(gray_frame.rows, gray_frame.cols, DataType<int>::type);
	thresholds(tmp_image4, tmp_image5);

	// 3. Detect Faces and Display Result
	//detectAndDisplay(frame, frame_output);

	// Circle detection
	detectCircles(tmp_image5, tmp_image5);

	// 4. Save Result Image
	imwrite( "detected.jpg", tmp_image5 );

	//visualise the loaded image in the window
	imwrite("convX.jpg", tmp_image1);
	imwrite("convY.jpg", tmp_image2);
	imwrite("direc.jpg", tmp_image3);
	imwrite("magni.jpg", tmp_image4);
	imwrite("thres.jpg", tmp_image5);


	return 0;
}


/** @function detectAndDisplay */
void detectAndDisplay( Mat &frame, Mat &output )
{
	std::vector<Rect> faces;
	Mat gray_image;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, gray_image, CV_BGR2GRAY );
	equalizeHist( gray_image, gray_image );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( gray_image, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

  // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(output, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}

void detectCircles(Mat &thres_frame, Mat &output )
{
	//Mat gray_image;

	/// Convert it to gray
  //cvtColor( thres_frame, gray_image, CV_BGR2GRAY );
	thres_frame.convertTo(thres_frame,CV_8UC1);
  /// Reduce the noise so we avoid false circle detection
  //GaussianBlur( gray_image, gray_image, Size(5, 5), 2, 2 );
	GaussianBlur( thres_frame, thres_frame, Size(9, 9), 2, 2 );

  vector<Vec3f> circles;

  /// Apply the Hough Transform to find the circles
  //HoughCircles( gray_image, circles, CV_HOUGH_GRADIENT, 1.2, gray_image.rows/12, 170, 100, 0, 0 );
	//HoughCircles( gray_image, circles, CV_HOUGH_GRADIENT, dp, minDist, param1, param2, minrad, maxrad );
	//HoughCircles( thres_frame, circles, CV_HOUGH_GRADIENT, 1.1, 30, 200, 70, 0, 0 );
	HoughCircles(thres_frame, circles, CV_HOUGH_GRADIENT, 1.2, 1, 200, 100, 0, 0);

	std::cout << circles.size() << std::endl;

  /// Draw the circles detected
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( output, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( output, center, radius, Scalar(0,25*i,255), 3, 8, 0 );
   }
}


void sobelConvolution(cv::Mat &gray_image, cv::Mat &output, int ker[], int image_num) {
	//define a temp image.
	Mat tmp_image = Mat::zeros(gray_image.rows, gray_image.cols, gray_image.type());
  //Mat int_tmp_image;
	// Threshold by looping through all pixels.
	for (int y = 1; y<gray_image.rows - 1; y++) {
		for (int x = 1; x<gray_image.cols - 1; x++) {
			/*if( (0<y<(image.rows-1)) && (0<x<(image.cols-1)) ) {*/
			uchar pixel1 = gray_image.at<uchar>(y - 1, x - 1);
			uchar pixel2 = gray_image.at<uchar>(y - 1, x);
			uchar pixel3 = gray_image.at<uchar>(y - 1, x + 1);
			uchar pixel4 = gray_image.at<uchar>(y, x - 1);
			uchar pixel5 = gray_image.at<uchar>(y, x);
			uchar pixel6 = gray_image.at<uchar>(y, x + 1);
			uchar pixel7 = gray_image.at<uchar>(y + 1, x - 1);
			uchar pixel8 = gray_image.at<uchar>(y + 1, x);
			uchar pixel9 = gray_image.at<uchar>(y + 1, x + 1);
			int pixel_val = (pixel1*ker[8] + pixel2*ker[7] + pixel3*ker[6] + pixel4*ker[5] + pixel5*ker[4] + pixel6*ker[3] + pixel7*ker[2] + pixel8*ker[1] + pixel9*ker[0]) / 9;

      output.at<int>(y, x) = cvRound(255*(max_val[image_num] - pixel_val) / (max_val[image_num] - min_val[image_num]) );
      //printf("%d\n", output.at<int>(y,x));
    }
	}
  //tmp_image.convertTo(int_tmp_image, CV_32F);
	//output = tmp_image;
}

void direction(cv::Mat &gray_image1, cv::Mat &gray_image2, cv::Mat &output) { // image1 is dx.
	int min = 0;
	int max = 0;
	for (int y = 0; y < gray_image1.rows; y++) {
		for (int x = 0; x < gray_image1.cols; x++) {
			int pixeldx = gray_image1.at<int>(y, x);
			int pixeldy = gray_image2.at<int>(y, x);
			float direct = (fastAtan2((float)pixeldy, (float)pixeldx));
			output.at<float>(y, x) = (direct/180)*pi;
			/*printf("%f\n", output.at<float>(y, x));*/
			/*if (output.at<float>(y, x) > 10) {
				printf("%f\n", output.at<float>(y, x));
			}*/
			//printf("%f\n", output.at<float>(y, x));
		/*	if (output.at<float>(y, x) < min) {
				min = output.at<float>(y, x);
			}
			if (output.at<float>(y, x) > max) {
				max = output.at<float>(y, x);
			}*/
		}
	}
}
void magnitude(cv::Mat &gray_image1, cv::Mat &gray_image2, cv::Mat &output) { // image1 is dx.
	int min = 0;
	int max = 0;
	for (int y = 0; y < gray_image1.rows; y++) {
		for (int x = 0; x < gray_image1.cols; x++) {
			int pixeldx = gray_image1.at<int>(y, x);
			int pixeldy = gray_image2.at<int>(y, x);
			int magni = cvRound(sqrt(pow(pixeldx, 2) + pow(pixeldy, 2)));
			output.at<int>(y, x) = cvRound(255 * (max_mag[0] - magni) / (max_mag[0] - min_mag[0]));
			//if (output.at<int>(y, x) > 0) {
			//	printf("%d\n", output.at<int>(y, x));
			//}
			//printf("%d\n", output.at<int>(y, x));
			if (output.at<int>(y, x) < min) {
				min = output.at<int>(y, x);
			}
			if (output.at<int>(y, x) > max) {
				max = output.at<int>(y, x);
			}
		}
	}
	//printf("min = %d\n", min); //0
	//printf("max = %d\n", max); //294
}

void thresholds(cv::Mat &gray_image, cv::Mat &output) {
	for (int y = 0; y < gray_image.rows; y++) {
		for (int x = 0; x < gray_image.cols; x++) {
		/*	if (output.at<int>(y, x) > 90) {
				printf("%d\n", output.at<int>(y, x));
			}*/
			if (gray_image.at<int>(y, x) > 90) {
				output.at<int>(y, x) = 255;
			}
			else {
				output.at<int>(y, x) = 0;
			}
		}
	}
}
