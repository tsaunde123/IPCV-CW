#include "functions.h"

// --------- for a <rectangle>, return its center point(for next function) --------------
Point rectangleCenter(Rect &rects) {
	return Point((rects.x + rects.width / 2), rects.y + rects.height / 2);
}

bool sortCircles(const cv::Vec3f &c1, const cv::Vec3f &c2) {
	return c1[2] > c2[2];
}

void houghCircles(Mat &output, Mat &thres, Mat &direc, vector<Vec3f> &circles, vector<cv::Point> &circleCenters) {
	// ----------------------In order to search between radius 20-100, we define 100 depth to get the correct index.
	int minRadius = 25;
	int threshold = 90;
	int depth = 150;
	int sizes[] = { thres.rows, thres.cols, depth };
	Mat hmat = Mat::zeros(3, sizes, CV_32SC1);
	Mat houghspace = Mat::zeros(thres.rows, thres.cols, DataType<int>::type);
	int x0, y0 = 0;

	clock_t time1, time2, time3, time4;
	double totaltime1, totaltime2, totaltime3;
	time1 = clock();

	for (int y = 0; y < thres.rows; y++) {
		for (int x = 0; x < thres.cols; x++) {
			for (int r = minRadius; r < depth; r++) {
				if (thres.at<uchar>(y, x) == 255) {

					x0 = x - (r * cos(direc.at<float>(y, x)));
					y0 = y - (r * sin(direc.at<float>(y, x)));

					if (x0 + r < thres.cols && x0 - r > 0 && y0 + r < thres.rows && y0 - r > 0) {
						hmat.at<int>(y0, x0, r) = hmat.at<int>(y0, x0, r) + 1;
					}

				}
			}
		}
	}

	// -------- turn 3d matrix to vector and calculate the max-------
	//reshape  - creates a new mat header of 1 row without copying data
	Mat result = hmat.reshape(0, 1);
	// declare vector and alloc size
	std::vector<int> outVector;
	outVector.reserve(result.cols);
	//copy data
	result.copyTo(outVector);
	int max = *max_element(begin(outVector), end(outVector));

	// -------------- take out all circle -----------------
	// std::vector<Vec3f> circles;
	int max_val = 0.9 * max;
	int circleNum = 0;

	time2 = clock();	
	for (int y = 0; y < thres.rows; y++) {
		for (int x = 0; x < thres.cols; x++) { //assuming input image will be gradiant image
			for (int r = minRadius; r< depth; r++) {
				// for(int r = depth-1; r>= minRadius; r--){
				if (hmat.at<int>(y, x, r) == max) {
					circleCenters.push_back(Point(x, y));
					circles.push_back(cv::Vec3f(x, y, r));
					circleNum++;
				}
			}
		}
	}

	time3 = clock();
	for(int y = 0; y < thres.rows; y++){
		for(int x = 0; x < thres.cols; x++){
		 	int sum_r = 0;
		 	for(int r=minRadius; r<depth; r++){
		 		sum_r = sum_r + hmat.at<int>(y, x, r);
		 	}
		 	houghspace.at<int>(y,x) = sum_r;
		}
	}
	normalization(houghspace);
	imwrite("circlehough.jpg", houghspace);

	time4 = clock();
	totaltime1 = (double)(time2 - time1) / CLOCKS_PER_SEC;
	totaltime2 = (double)(time3 - time2) / CLOCKS_PER_SEC;
	totaltime3 = (double)(time4 - time2) / CLOCKS_PER_SEC;
}

// --------- for circles have been detected, remove overlaped ones ------------
void removeOverlapCircle(vector<Vec3f> &sortedCircles, vector<Vec3f> &outputCircles) {
	outputCircles.push_back(sortedCircles[0]);
	for (int i = 1; i<sortedCircles.size(); i++) {
		Point a = (sortedCircles[i][0], sortedCircles[i][1]);
		bool validCircle = true;

		for (int j = 0; j<outputCircles.size(); j++) {
			Point b = (outputCircles[j][0], outputCircles[j][1]);
			double RadiusDifference = fabs(outputCircles[j][2] - sortedCircles[i][2]);
			if ((cv::norm(a - b) <= RadiusDifference) || ((cv::norm(a - b) - RadiusDifference) < 10)) {
				validCircle = validCircle && false;
			}
			else {
				validCircle = validCircle && true;
			}
		}

		if (validCircle) {
			outputCircles.push_back(sortedCircles[i]);
		}
	}
}

// --------- for each circle, remove overlap rectangle above it, and return the index of the best ------------
int removeOverlapRect(std::vector<Vec3f> &relateRectsCenter, cv::Point circle) {
	int nearestRectIndex = 0;
	int shortestDistance = 10000;
	for (int i = 0; i < relateRectsCenter.size(); i++) {
		Point relateRectCenter(relateRectsCenter[i][0], relateRectsCenter[i][1]);
		if (cv::norm(relateRectCenter - circle) < shortestDistance) {
			shortestDistance = cv::norm(relateRectCenter - circle);
			nearestRectIndex = relateRectsCenter[i][2];
		}
	}
	return nearestRectIndex;
}
