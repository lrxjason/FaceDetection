
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetect.h"

#pragma comment(lib,"libfacedetect.lib")


//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	
	cout << "start camera " << endl;
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;
		cout << "camera initial good " << endl;

	//Read the video stream
		while (true)
		{
			Mat image;
			cap >> image; // get a new frame from camera

			Mat gray;
			cvtColor(image, gray, CV_BGR2GRAY);


			int * pResults = NULL;
			//pBuffer is used in the detection functions.
			//If you call functions in multiple threads, please create one buffer for each thread!
			unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
			if (!pBuffer)
			{
				fprintf(stderr, "Can not alloc buffer.\n");
				return -1;
			}

			int doLandmark = 0;

			pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
				1.2f, 3, 48, 0, doLandmark);

			printf("%d faces detected.\n", (pResults ? *pResults : 0));
			Mat result_multiview_reinforce = image.clone();;
			//print the detection results
			for (int i = 0; i < (pResults ? *pResults : 0); i++)
			{
				short * p = ((short*)(pResults + 1)) + 142 * i;
				int x = p[0];
				int y = p[1];
				int w = p[2];
				int h = p[3];
				int neighbors = p[4];
				int angle = p[5];

				printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
				rectangle(result_multiview_reinforce, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
				if (doLandmark)
				{
					for (int j = 0; j < 68; j++)
						circle(result_multiview_reinforce, Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, Scalar(0, 255, 0));
				}
			}
			string name = "FaceDetection";
			imshow(name, result_multiview_reinforce);
			char c=waitKey(1);
			if (c == 27)
			{
				break;
			}

			////release the buffer
			free(pBuffer);
		}
	return 0;
}