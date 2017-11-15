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

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat &frame, Mat &output );
void detectCircles(Mat &frame, Mat &output);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat frame_output = frame;

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay(frame, frame_output);

	// Circle detection
	detectCircles(frame, frame_output);

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

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

void detectCircles(Mat &frame, Mat &output )
{
	Mat gray_image;
	int rows;
	int cols;
	cv::Size size = gray_image.size();
	rows = size.height;
	cols = size.width;

	/// Convert it to gray
  cvtColor( frame, gray_image, CV_BGR2GRAY );

  /// Reduce the noise so we avoid false circle detection
  //GaussianBlur( gray_image, gray_image, Size(5, 5), 2, 2 );
	GaussianBlur( gray_image, gray_image, Size(9, 9), 2, 2 );

  vector<Vec3f> circles;

  /// Apply the Hough Transform to find the circles
  //HoughCircles( gray_image, circles, CV_HOUGH_GRADIENT, 1.2, gray_image.rows/12, 170, 100, 0, 0 );
	//HoughCircles( gray_image, circles, CV_HOUGH_GRADIENT, dp, minDist, param1, param2, minrad, maxrad );
	//HoughCircles( gray_image, circles, CV_HOUGH_GRADIENT, 1.1, 30, 200, 70, 0, 0 );
	HoughCircles(gray_image, circles, CV_HOUGH_GRADIENT, 1, gray_image.rows/10, 100, 40, 30, 50);

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
