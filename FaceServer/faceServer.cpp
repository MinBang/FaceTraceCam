#include <iostream>
#include <WinSock2.h>
#include <vector>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "Lib/SerialComm.h"
#include "NetLib\NetLib.h"
#include "Lib/FaceRec.h"

#pragma comment(lib, "ws2_32")

#define WINDOWS_X 320
#define	WINDOWS_Y 240

#define FORWARD  'V'
#define BACKWARD 'H'

#define MAXSIZE 90

#define BUF_SIZE 50

using namespace std;
using namespace cv;

////////////////////////////

int SendServoSerial(char data);
int OpenSerialConn();
int CloseSerialConn();
int RunImage();
int MoveSubo(int midFaceX, int midFaceY, int& servoHPosition, int& servoVPosition);
int MoveSubo(int servoHPosition, int servoVPosition);
void SendState(char servoHPosition, char servoVPosition);
void RunVideo();
void SplitMessage(vector<char*>& msgs, char* message);
void HandleMessage(char* message);
void IniCamState();
int RunServer(SOCKET s);

////////////////////

typedef struct camState
{
	bool isAuto;
	bool isUse;

	SOCKET s;
}S_CAMSTATE;

String face_cascade = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml";
String eye_cascade = "C:/opencv/sources/data/haarcascades/haarcascade_eye.xml";

S_CAMSTATE* camState = NULL;

int midScreenY = (WINDOWS_Y / 2);
int midScreenX = (WINDOWS_X / 2);
int midScreenWindow = 30; //화면 중앙에서 어느정도 위치안에 얼굴의 중앙위치점가 들어올 경우 
//스크린에서 중앙으로 들어왔다고 인식할 것인지 오차범위 지정
int servoHPosition = 90;
int servoVPosition = 90;

bool connectState;

unsigned char stepSize = 3;
unsigned char CharCount = 0; 

CSerialComm serialComm; //SerialComm 객체 생성

int TestServer(SOCKET s)
{
	int len = 0;
	char buf[BUF_SIZE] = {0, };

	ZeroMemory(buf, sizeof(buf));

	len = recv(s, buf, sizeof(buf), 0);

	printf("len : %d - msg : %s\n", len, buf);

	ZeroMemory(buf, sizeof(buf));
	strcpy(buf, "hello\0");

	int re = send(s, buf, strlen(buf), 0);

	printf("re Send : %d\n", re);

	return 0;
}


int RunServer(SOCKET s)
{
	int len = 0;
	char buf[BUF_SIZE] = {0, };

	len = recv(s, buf, sizeof(buf), 0);

	if(0 < len)
	{
		printf("len : %d - msg : %s\n", len, buf);

		if(camState->s == NULL)
		{
			camState->s = s;
		}

		HandleMessage(buf);
	}

	if(s != camState->s)
	{
		closesocket(s);
	}	

	return 0;
}

void imagesend(SOCKET s, SOCKADDR_IN raddr, Mat frame)
{
	IplImage* image;
	IplImage* dst_img;

	CvMat* l_buf;

	dst_img = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);	//cvCreateImage는 IplImage 구조체의 메모리를 생성하여 그 포인터를 넘겨준다.
	//1번째 크기, 2번째 깊이, 3번째 채널 수(1 == GRAY, 3 == RGB)

	//영상의 용량을 줄이기 위해 해상도 320*240 변경
	image = &IplImage(frame);
	cvResize(image, dst_img, CV_INTER_LINEAR);

	//영상을 내부API를 이용하여 jpg로 인코딩
	l_buf = cvEncodeImage(".jpg", dst_img, 0);

	int retval = sendto(s, (const char*)l_buf->data.ptr, l_buf->step, 0, (SOCKADDR*)&raddr, sizeof(raddr));
	if (retval == SOCKET_ERROR)
	{
		perror("sendto");
	}

	//printf("%d바이트를 보냈습니다.\n", retval);
	
	cvReleaseMat(&l_buf);
	cvReleaseImage(&dst_img);
}

int  FaceRecognitionEX(){

	int time = 0;
	
	cout << "start recognizing..." << endl;

	//load pre-trained data sets
	Ptr<FaceRecognizer>  model = createFisherFaceRecognizer();
	model->load("image/fisherface.yml");

	Mat testSample = imread("image/bang1.jpg", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;


	//lbpcascades/lbpcascade_frontalface.xml
	string classifier = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";

	CascadeClassifier face_cascade;
	string window = "Capture - face detection";

	if (!face_cascade.load(classifier)){
		cout << " Error loading file" << endl;
		return -1;
	}

	VideoCapture cap(0);
	//VideoCapture cap("C:/Users/lsf-admin/Pictures/Camera Roll/video000.mp4");

	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return -1;
	}

	//double fps = cap.get(CV_CAP_PROP_FPS);
	//cout << " Frames per seconds " << fps << endl;
	namedWindow(window, 1);
	long count = 0;

	//////////////////////////
	//////////////////////////

	SOCKET s = socket(AF_INET, SOCK_DGRAM, 0);
	if (INVALID_SOCKET == s)
	{
		printf("broadcast socket error!!!\n");
		exit(1);
	}

	SOCKADDR_IN raddr;
	ZeroMemory(&raddr, sizeof(raddr));
	raddr.sin_family = AF_INET;
	raddr.sin_addr.s_addr = htonl(INADDR_ANY);  // 1.209.148.255
	raddr.sin_port = htons(2015);

	bind(s, (SOCKADDR*)&raddr, sizeof(raddr));
	
	SOCKADDR_IN caddr;
	int len = sizeof(caddr);
	char buf[BUF_SIZE] = {0, };

	recvfrom(s, buf, sizeof(buf), 0, (SOCKADDR*)&caddr, &len);

	/////////////////////////
	/////////////////////////

	while (true)
	{
		if(camState->isUse == false)
		{
			break;
		}

		time++;
		if(20 <= time && camState->isAuto == true)
		{
			time = 0;
			SendState(servoHPosition, servoVPosition);
		}

		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;
		Rect face_i;

		cap >> frame;
		//cap.read(frame);
		count = count + 1;//count frames;

		if (!frame.empty()){

			//clone from original frame
			original = frame.clone();

			cv::resize(original, original, cv::Size(WINDOWS_X,
				WINDOWS_Y), 0, 0, CV_INTER_NN); // downsample 1/2x

			//convert image to gray scale and equalize
			cvtColor(original, graySacleFrame, CV_BGR2GRAY);
			//equalizeHist(graySacleFrame, graySacleFrame);

			

			//detect face in gray image
			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));


			//number of faces detected
			//cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			//region of interest
			//cv::Rect roi;

			//person name
			string Pname = "";

			if(0 < faces.size())
			{
				face_i = faces[0];

				for( int i=1 ; i < faces.size() ; i++)
				{
					if((face_i.width * face_i.height) < (faces[i].width * faces[i].height))
					{
						face_i = faces[i];
					}
				}
			


			//for (int i = 0; i < faces.size(); i++)
			//{
			//	face_i = faces[i];

				//crop the roi from grya image
				Mat face = graySacleFrame(face_i);

				//resizing the cropped image to suit to database image sizes
				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

				//recognizing what faces detected
				int label = -1; double confidence = 0;
				model->predict(face_resized, label, confidence);

				//cout << "label : " << label << endl;

				//cout << " confidencde " << confidence << endl;

				//drawing green rectagle in recognize face
				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
				/*string text = "Detected";
				if (label == 0){
					//string text = format("Person is  = %d", label);
					Pname = "bang";
				}
				else if(label == 1)
				{
					Pname = "donghyuk";
				}
				else if(label == 2)
				{
					Pname = "junhyuk";
				}
				else if(label == 3)
				{
					Pname = "jungwoo";
				}
				else if(label == 4)
				{
					Pname = "onejun";
				}
				else if(label == 5)
				{
					Pname = "ryu";
				}
				else{
					Pname = "unknown";
				}

				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);

				//name the person who is in the image
				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);*/
				//cv::imwrite("E:/FDB/"+frameset+".jpg", cropImg);
			}


			//putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			//putText(original, "Person: " + Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			//display to the winodw
			cv::imshow(window, original);

			//cout << "model infor " << model->getDouble("threshold") << endl;

			imagesend(s, caddr, original);

			if(true == camState->isAuto)
			{
				if (0 < faces.size())
				{
					int midFaceX = faces[0].x + (faces[0].width / 2);
					int midFaceY = faces[0].y + (faces[0].height / 2);

					//cout << "midFaceX : " << midFaceX << endl;
					//cout << "midFaceY : " << midFaceY << endl;

					MoveSubo(midFaceX, midFaceY, servoHPosition, servoVPosition);
				}
			}

		}
		if (waitKey(50) >= 0) break;
	}

	camState->isUse = false;
	camState->isAuto = false;

	cap.release();
	cv::destroyWindow(window);
}

int main()
{
	////////////////////////
	//fisherFaceTrainer();

	//FaceRecognitionEX();
	///////////////////////////

	//RunVideo();

	IniCamState();

	//이거
	if(0 == OpenSerialConn())
	{
		connectState = true;
	}

	/*for(int i=0;126;i++)
	{
		MoveSubo(i, i);

		Sleep(15);
	}*/

	IniTCPServerM(5050, RunServer);

	//이거

	if(true == connectState)
	{
		CloseSerialConn();
	}	

	//IniTCPServerM(5050, TestServer);
	
	return 0;
}

void IniCamState()
{
	camState = new S_CAMSTATE;
	camState->isAuto = false;
	camState->isUse = false;

	camState->s = NULL;

	connectState = false;
}

/*
	- Serial 함수
*/

int SendServoSerial(char data)
{
	if (!serialComm.sendCommand(data))
	{
		cout << "send command failed" << endl;
	}
	else
	{
		//cout << "send Command success" << endl;
		//cout << (int)data << "     " << data << endl;
	}

	return 0;
}

int OpenSerialConn()
{
	if (!serialComm.connect("COM4")) //COM25번의 포트를 오픈한다. 실패할 경우 -1을 반환한다.
	{
		cout << "connect faliled" << endl;
		return -1;
	}
	else
	{
		cout << "connect successed" << endl;
	}

	return 0;
}

int CloseSerialConn()
{
	serialComm.disconnect(); //작업이 끝나면 포트를 닫는다

	cout << "end connect" << endl;

	return 0;
}

/*
	- OpenCV를 이용한 함수
*/

int RunImage()
{
	CascadeClassifier face;
	CascadeClassifier eye;

	Mat img = imread("C:/opencv/images/front.jpg");

	if (img.data == NULL)
	{
		cout << "이미지 열기 실패" << endl;

		return -1;
	}

	if (!face.load(face_cascade) || !eye.load(eye_cascade))
	{
		cout << "cascade 파일 load 실패" << endl;

		return -1;
	}

	Mat gray;

	cvtColor(img, gray, CV_RGB2GRAY);

	vector<Rect> face_pos;
	face.detectMultiScale(gray, face_pos, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

	for (int i = 0; i<(int)face_pos.size(); i++)
	{
		rectangle(img, face_pos[i], Scalar(0, 255, 0), 2);
	}

	////////////////////////////////////////

	for (int i = 0; i<(int)face_pos.size(); i++)
	{
		vector<Rect> eye_pos;

		Mat roi = gray(face_pos[i]);
		eye.detectMultiScale(roi, eye_pos, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

		for (int j = 0; j<(int)eye_pos.size(); j++)
		{
			Point center(face_pos[i].x + eye_pos[j].x + (eye_pos[j].width / 2),
				face_pos[i].y + eye_pos[j].y + (eye_pos[i].height / 2));

			int radius = cvRound((eye_pos[j].width + eye_pos[j].height) * 0.2);
			circle(img, center, radius, Scalar(0, 0, 255), 2);
		}
	}

	namedWindow("test");
	namedWindow("test2");
	imshow("test", img);
	imshow("test2", gray);

	waitKey(0);

	return 0;
}

void RunVideo()
{
	int time = 0;

	// -------------------------------------------------------------------------
	// webcam routine
	
	cv::VideoCapture capture(0);

	capture.get(0);

	if (!capture.isOpened()) {
		std::cerr << "Could not open camera" << std::endl;
		return ;
	}

	// create a window
	cv::namedWindow("webcam");
	
	// -------------------------------------------------------------------------
	// face detection configuration
	cv::CascadeClassifier face_classifier;
	face_classifier.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml");

	while (true) {

		if(camState->isUse == false)
		{
			break;
		}

		bool frame_valid = true;
		time++;
		if(20 <= time && camState->isAuto == true)
		{
			time = 0;
			SendState(servoHPosition, servoVPosition);
		}

		cv::Mat frame_original;
		cv::Mat frame;

		try {
			capture >> frame_original; // get a new frame from webcam
			imwrite("NowFace/test.bmp", frame_original);

			cv::resize(frame_original, frame, cv::Size(320,
				240), 0, 0, CV_INTER_NN); // downsample 1/2x
		}
		catch (cv::Exception& e) {
			std::cerr << "Exception occurred. Ignoring frame... " << e.err
				<< std::endl;
			frame_valid = false;
		}

		if (frame_valid) {
			try {
				// convert captured frame to gray scale & equalize
				cv::Mat grayframe;
				cv::cvtColor(frame, grayframe, CV_BGR2GRAY);
				cv::equalizeHist(grayframe, grayframe);

				// -------------------------------------------------------------
				// face detection routine

				// a vector array to store the face found
				std::vector<cv::Rect> faces;

				face_classifier.detectMultiScale(grayframe, faces,
					1.1, // increase search scale by 10% each pass
					3,   // merge groups of three detections
					CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
					cv::Size(30, 30));

				// -------------------------------------------------------------
				// draw the results

				if(0 < faces.size())
				{
					Rect face_i = faces[0];

					for (int i = 0; i < faces.size(); i++)
					{

						if((face_i.width * face_i.height) < (faces[i].width * faces[i].height))
						{
							face_i = faces[i];
						}					
					}

					cv::Point lb(face_i.x + face_i.width, face_i.y + face_i.height);
					cv::Point tr(face_i.x, face_i.y);

					cv::rectangle(frame, lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);
				}

				// print the output
				cv::imshow("webcam", frame);

				if(true == camState->isAuto)
				{
					if (0 < faces.size())
					{
						int midFaceX = faces[0].x + (faces[0].width / 2);
						int midFaceY = faces[0].y + (faces[0].height / 2);

						//cout << "midFaceX : " << midFaceX << endl;
						//cout << "midFaceY : " << midFaceY << endl;

						MoveSubo(midFaceX, midFaceY, servoHPosition, servoVPosition);
					}
				}
			}
			catch (cv::Exception& e) {
				std::cerr << "Exception occurred. Ignoring frame... " << e.err
					<< std::endl;
			}
		}
		if (cv::waitKey(50) >= 0)
		{
			break;
		}
	}
	// VideoCapture automatically deallocate camera object

	camState->isUse = false;
	camState->isAuto = false;

	capture.release();
	cv::destroyWindow("webcam");
}

/*
	- 현재 얼굴의 위치를 추적하는 함수
*/

int MoveSubo(int servoHPosition, int servoVPosition)
{
	if(126 < servoHPosition && servoHPosition < 0)
	{
		return -1;
	}

	else if(126 < servoVPosition && servoVPosition < 0)
	{
		return -1;
	}

	if(false == connectState)
	{
		return -1;
	}

	SendServoSerial(FORWARD);
	SendServoSerial(servoVPosition);
	SendServoSerial(BACKWARD);
	SendServoSerial(servoHPosition);

	//SendState(servoHPosition, servoVPosition);

	return 0;
}

int MoveSubo(int midFaceX, int midFaceY, int& servoHPosition, int& servoVPosition)
{
	

	if (midFaceY < (midScreenY - midScreenWindow))
   {
      if (servoVPosition >= 5)
      {
         servoVPosition -= stepSize;
      }
   }

   //현재 얼굴의 위치가 스크린의 중앙보다 위에 위치할 경우 수직으로 움직이는 서보모터의 각도를 1도씩 증가시킨다
   else if (midFaceY >(midScreenY + midScreenWindow))
   {
      servoVPosition += stepSize;
      if (126 < servoVPosition)
      {
         servoVPosition = 126;
      }
   }
   else
   {
      //cout << "높이는 중앙 ";
   }

   //현재 얼굴의 위치가 스크린의 중앙보다 왼쪽에 위치할 경우 수평으로 움직이는 서보모터의 각도를 1도씩 감소시킨다
   if (midFaceX < (midScreenX - midScreenWindow))
   {
      servoHPosition += stepSize;
      if (126 < servoHPosition)
      {
         servoHPosition = 126;
      }
      
   }

   //현재 얼굴의 위치가 스크린의 중앙보다 아래에 위치할 경우 수평으로 움직이는 서보모터의 각도를 1도씩 증가시킨다
   else if (midFaceX > midScreenX + midScreenWindow)
   {
      if (servoHPosition >= 5)
      {
         servoHPosition -= stepSize;
      }
   }
   else
   {
      //cout << "수평은 중앙" << endl;
   }

   if(false == connectState)
	{
		return -1;
	}

   //cout << "보내는중" << endl;

   SendServoSerial(FORWARD);
   SendServoSerial(servoVPosition);
   SendServoSerial(BACKWARD);
   SendServoSerial(servoHPosition);

   //SendState(servoHPosition, servoVPosition);

	//SendState(servoHPosition, servoVPosition);

	return 0;
}

/*
	- Mobius에 Data를 업로드 하는 함수
*/

void SendState(char servoHPosition, char servoVPosition)
{
	char state[5] = {0,};

	state[0] = servoHPosition;
	state[1] = ',';
	state[2] = servoVPosition;
	state[3] = ',';

	cout << "servo : " << (int)servoHPosition << " " << (int)servoVPosition << endl;
	
	int re = send(camState->s, (char *)state, strlen((char *)state), 0);

	printf("re Send : %d\n", re);
}

//////////////////////////////////////////////////////

// ',' 단위로 split해주는 함수
void SplitMessage(vector<char*>& msgs, char* message)
{
	char buf[10] = {0,};
	int start = 0;
	int len = strlen(message);
	int index = 0;
	int end = 0;

	for(int i=0;i<len;i++)
	{
		if(message[i] == ',' || i == (len - 1))
		{
			if(10 <= (i-start))
			{
				cout << "vector 삽입 범위초과" << endl;

				break;
			}

			index = 0;
			end = i;

			if(i == (len - 1))
			{
				end = len;
			}

			for(int j=start;j<end;j++)
			{
				buf[index] = message[j];

				index++;
			}

			buf[index] = '\0';
			start = i+1;

			char* pushBuf = new char[strlen(buf)+1];
			strcpy(pushBuf, buf);

			msgs.push_back(pushBuf);
		}
	}
}

// face서버로 들어오면 요청을 handling해주는 함수
void HandleMessage(char* message)
{
	vector<char*> msgs;
	SplitMessage(msgs, message);

	int len = msgs.size();

	if(len == 1)
	{
		//RunCam = Video;

		if(0 == strcmp(msgs[0], "auto"))
		{
			cout << "auto" << endl;

			if(camState->isUse == false)
			{
				camState->isUse = true;
				camState->isAuto = true;

				//RunVideo();
				FaceRecognitionEX();
			}
			else
			{
				camState->isAuto = true;
				cout << "camera가 실행중입니다" << endl;
			}
		}
		else if(0 == strcmp(msgs[0], "manual"))
		{
			cout << "manual" << endl;

			if(camState->isUse == false)
			{
				camState->isUse = true;
				camState->isAuto = false;

				//RunVideo();
				FaceRecognitionEX();
			}
			else
			{
				camState->isAuto = false;
				cout << "camera가 실행중입니다" << endl;
			}
			
		}
		else if(0 == strcmp(msgs[0], "close"))
		{
			cout << "close" << endl;

			camState->isUse = false;
			camState->isAuto = false;

			//closesocket(camState->s);
		}
		else if(0 == strcmp(msgs[0], "test"))
		{
			cout << "test" << endl;

			SendState(7, 7);

			servoHPosition = 60;
			servoVPosition = 40;

			//MoveSubo(servoHPosition, servoVPosition);
		}
		else if(0 == strcmp(msgs[0], "exit"))
		{
			exit(0);
		}
	}
	else if(len == 3)
	{
		if(0 == strcmp(msgs[0], "angle") && camState->isAuto == false)
		{
			int h = atoi(msgs[1]);
			int v = atoi(msgs[2]);

			cout << "angle : " << h << ", " << v << endl;

			servoHPosition = h;
			servoVPosition = v;

			if(0 == MoveSubo(h, v))
			{
				SendState(h, v);
			}
		}
	}
}

//////////////////////////////////////////////////////
