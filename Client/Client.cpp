// Client.cpp : �ܼ� ���� ���α׷��� ���� �������� �����մϴ�.
//

#include "stdafx.h"
#include <WinSock2.h>
#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>

#pragma comment(lib, "ws2_32.lib")
#define BUFSIZE 300000

typedef struct IMAGE
{
	CvMat* b_buf;
	IplImage * image;
};

IMAGE m_pImage = { NULL };


int _tmain(int argc, _TCHAR* argv[])
{
	WSADATA wsa;

	if (0 != WSAStartup(MAKEWORD(2, 2), &wsa))
	{
		printf("���� �ʱ�ȭ ����!!\n");
		return -1;
	}

	SOCKET s = socket(AF_INET, SOCK_DGRAM, 0);

	if (INVALID_SOCKET == s)
	{
		printf("invalid socket!!\n");
		return -1;
	}

	SOCKADDR_IN caddr;
	ZeroMemory(&caddr, sizeof(caddr));
	caddr.sin_family = AF_INET;
	caddr.sin_addr.s_addr = inet_addr("192.168.43.154");
	caddr.sin_port = htons(2015);

	//int re = bind(s, (SOCKADDR*)&caddr, sizeof(caddr));

	//if (SOCKET_ERROR == re)
	{
		//printf("bind error!!\n");
		//return -1;
	}

	char* title = "test";
	cvNamedWindow(title, CV_WINDOW_AUTOSIZE);

	int addrlen, retval;
	SOCKADDR_IN saddr;
	char buf[BUFSIZE + 1];

	int re;

	sendto(s, (char*)&re, sizeof(re), 0, (SOCKADDR*)&caddr, sizeof(caddr));

	while (1)
	{
		addrlen = sizeof(saddr);
		retval = recvfrom(s, buf, BUFSIZE, 0, (SOCKADDR *)&saddr, &addrlen);
		if (retval == SOCKET_ERROR)
		{
			perror("recvfrom");
			continue;
		}

		//���� ���ڵ� ����
		m_pImage.b_buf = cvCreateMatHeader(1, 1, 1111638016);
		m_pImage.b_buf->data.ptr = (uchar*)buf;
		m_pImage.b_buf->step = retval;
		m_pImage.b_buf->cols = m_pImage.b_buf->step;
		m_pImage.b_buf->width = m_pImage.b_buf->step;
		m_pImage.b_buf->refcount = (int*)0x03858480;

		//���� ���ڵ�API�� ����Ͽ� ���� �������� ����
		m_pImage.image = cvDecodeImage(m_pImage.b_buf);

		//���� ������ ���
		buf[retval] = '\0';
		printf("[UDP / %s : %d] \t %d\n", inet_ntoa(saddr.sin_addr), ntohs(saddr.sin_port), retval);

		cvShowImage(title, m_pImage.image);

		if (cvWaitKey(10) == 27)		//esc ������ ������
			break;
	}

	cvReleaseMat(&m_pImage.b_buf);
	cvDestroyWindow(title);
	cvReleaseImage(&m_pImage.image);

	return 0;

	WSACleanup();

	return 0;
}

