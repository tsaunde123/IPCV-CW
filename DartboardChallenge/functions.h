#ifndef FUNCTIONS_H
#define FUNCTIONS_H

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
#define pi 3.1415926

using namespace cv;
using namespace std;

void normalization(cv::Mat &gray_image);

Point rectangleCenter(Rect &rects);

bool sortCircles(const cv::Vec3f &c1, const cv::Vec3f &c2);

void houghCircles(Mat &output, Mat &thres, Mat &direc, vector<Vec3f> &circles, vector<cv::Point> &circleCenters);

void removeOverlapCircle(vector<Vec3f> &sortedCircles, vector<Vec3f> &outputCircles);

int removeOverlapRect(std::vector<Vec3f> &relateRectsCenter, cv::Point circle);

void groundTruth(vector<vector<Rect>> &groundtruth);

float f1_score(vector<Rect> &validrects, Mat image, int imageIndex, vector<vector<Rect>> &groundtruth);

bool isIntersects(Rect rect1, Rect rect2);

#endif
