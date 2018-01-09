#include "functions.h"

void groundTruth(vector<vector<Rect>> &groundtruth) {
	Rect face14_1(450, 200, 123, 143);
	Rect face14_2(706, 169, 140, 140);
	vector<Rect> face14;
	face14.push_back(face14_1);
	face14.push_back(face14_2);
	//groundtruth.push_back(face14);

	Rect image0_1(400, 0, 235, 220);
	vector<Rect> image0;
	image0.push_back(image0_1);
	groundtruth.push_back(image0);

	Rect image1_1(150, 80, 290, 290);
	vector<Rect> image1;
	image1.push_back(image1_1);
	groundtruth.push_back(image1);

	Rect image2_1(40, 40, 200, 200);
	vector<Rect> image2;
	image2.push_back(image2_1);
	groundtruth.push_back(image2);

	Rect image3_1(300, 125, 110, 115);
	vector<Rect> image3;
	image3.push_back(image3_1);
	groundtruth.push_back(image3);

	Rect image4_1(140, 50, 280, 280);
	vector<Rect> image4;
	image4.push_back(image4_1);
	groundtruth.push_back(image4);

	Rect image5_1(400, 120, 170, 160);
	vector<Rect> image5;
	image5.push_back(image5_1);
	groundtruth.push_back(image5);


	Rect image6_1(190, 100, 100, 100);
	vector<Rect> image6;
	image6.push_back(image6_1);
	groundtruth.push_back(image6);

	Rect image7_1(225, 130, 205, 220);
	vector<Rect> image7;
	image7.push_back(image7_1);
	groundtruth.push_back(image7);

	Rect image8_1(50, 230, 100, 140);
	Rect image8_2(810, 180, 180, 200);
	vector<Rect> image8;
	image8.push_back(image8_1);
	image8.push_back(image8_2);
	groundtruth.push_back(image8);

	Rect image9_1(150, 5, 315, 325);
	vector<Rect> image9;
	image9.push_back(image9_1);
	groundtruth.push_back(image9);

	Rect image10_1(25, 35, 225, 245);
	Rect image10_2(530, 70, 140, 200);
	Rect image10_3(880, 100, 110, 160);
	vector<Rect> image10;
	image10.push_back(image10_1);
	image10.push_back(image10_2);
	image10.push_back(image10_3);
	groundtruth.push_back(image10);

	Rect image11_1(150, 80, 100, 120);
	vector<Rect> image11;
	image11.push_back(image11_1);
	groundtruth.push_back(image11);

	Rect image12_1(90, 20, 180, 240);
	vector<Rect> image12;
	image12.push_back(image12_1);
	groundtruth.push_back(image12);

	Rect image13_1(210, 50, 250, 270);
	vector<Rect> image13;
	image13.push_back(image13_1);
	groundtruth.push_back(image13);

	Rect image14_1(60, 30, 260, 260);
	Rect image14_2(915, 30, 260, 260);
	vector<Rect> image14;
	image14.push_back(image14_1);
	image14.push_back(image14_2);
	groundtruth.push_back(image14);

	Rect image15_1(120, 30, 190, 190);
	vector<Rect> image15;
	image15.push_back(image15_1);
	groundtruth.push_back(image15);

}

float f1_score(vector<Rect> &validrects, Mat image, int imageIndex, vector<vector<Rect>> &groundtruth) {

	for (int i = 0; i < groundtruth[imageIndex].size(); i++)
	{
		rectangle(image, Point(groundtruth[imageIndex][i].x, groundtruth[imageIndex][i].y),
			Point(groundtruth[imageIndex][i].x + groundtruth[imageIndex][i].width, groundtruth[imageIndex][i].y + groundtruth[imageIndex][i].height), Scalar(0, 255, 255), 2);
	}
	int TP = 0;
	int FP = 0;
	int FN = 0;
	for (int i = 0; i < validrects.size(); i++) {
		bool Notvalid = true;
		for (int j = 0; j < groundtruth[imageIndex].size(); j++) {
			if (isIntersects(validrects[i], groundtruth[imageIndex][j])) {
				float areaIntersect = (float)(validrects[i] & groundtruth[imageIndex][j]).area();
				float areaBox = (float)validrects[i].area();
				if (((float)((validrects[i] & groundtruth[imageIndex][j]).area()) / (float)validrects[i].area()) > 0.5) {
					Notvalid = Notvalid && false;
					TP++;
				}
				else {
					Notvalid = Notvalid && true;
				}
			}
			else {
				Notvalid = Notvalid && true;
			}
		}
		if (Notvalid) {
			FP++;
		}

	}
	for (int i = 0; i < groundtruth[imageIndex].size(); i++) {
		bool missedTruth = true;
		for (int j = 0; j < validrects.size(); j++) {
			if (isIntersects(validrects[j], groundtruth[imageIndex][i])) {
				float areaIntersect = (float)(validrects[j] & groundtruth[imageIndex][i]).area();
				float areaBox = (float)validrects[j].area();
				if ((areaIntersect / areaBox) > 0.5) {
					missedTruth = missedTruth && false;
				}
				else {
					missedTruth = missedTruth && true;
				}
			}
			else {
				missedTruth = missedTruth && true;
			}
		}
		if (missedTruth) {
			FN++;
		}
	}

	cout << "TP: " << TP << endl;
	cout << "FP: " << FP << endl;
	cout << "FN: " << FN << endl;
	float P = 0;
	float R = 0;
	float F1 = 0;
	if ((float)(TP + FP) == 0) {
		P = 0;
	}
	else {
		P = (float)TP / (float)(TP + FP);
	}

	if ((float)(TP + FN) == 0) {
		R = 0;
	}
	else {
		R = (float)TP / (float)(TP + FN);
	}
	
	if ((P + R) == 0) {
		F1 = 0;
	}
	else {
		F1 = 2 * P*R / (P + R);
	}


	//float P = (float)TP / (float)(TP + FP);
	//float R = (float)TP / (float)(TP + FN);
	//float F1 = 2 * P*R / (P + R);

	cout << "P: " << P << endl;
	cout << "R: " << R << endl;
	cout << "F1: " << F1 << endl;
	return F1;

}

bool isIntersects(Rect rect1, Rect rect2) {
	return ((rect1 & rect2).area() > 0);
}