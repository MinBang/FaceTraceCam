//#include "stdafx.h"
#include "NetLib.h"

#include <stdlib.h>
#include <stdio.h>
//---------------------------------------------------------------------
int recvn(SOCKET s, char* buf, int len, int flags)
{
	int received = 0;
	int     left = 0;
	char*    ptr = NULL;

	ptr = buf;
	left = len;

	while(0 < left)
	{
		received = recv(s, ptr, left, flags);
		if(SOCKET_ERROR == received)
		{
			return SOCKET_ERROR;
		}
		else if(0 == received)
		{
			break;
		}

		left -= received;
		ptr += received;
	}

	return len - left;
}
//---------------------------------------------------------------------


struct S_PARA 
{
	SOCKET cs;
	SOCKADDR_IN caddr;
};

int(*ServerProc)(SOCKET c) = NULL;

DWORD WINAPI ServerThread(LPVOID para)
{
	S_PARA* pPara = (S_PARA*)para;
	
	ServerProc(pPara->cs);

	//closesocket(pPara->cs);

	delete pPara;

	return 0;
}

//---------------------------------------------------------------------

int IniTCPServerM(int port, int(*proc)(SOCKET c))
{
	int ret = 0;
	WSADATA wsa;
	ServerProc = proc;

	ret = WSAStartup(MAKEWORD(2, 2), &wsa);
	if(0 != ret)
	{
		printf("WinSock Error...\n");
		return -1;
	}

	SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
	if(INVALID_SOCKET == s)
	{
		printf("Socket Error...\n");
		return -1;
	}

	SOCKADDR_IN saddr;
	ZeroMemory(&saddr, sizeof(saddr));

	saddr.sin_family = AF_INET;
	saddr.sin_addr.s_addr = htonl(INADDR_ANY);
	saddr.sin_port = htons(port);

	ret = bind(s, (SOCKADDR*)&saddr, sizeof(saddr));
	if(SOCKET_ERROR == ret)
	{
		printf("Bind Error...\n");
		return -1;
	}

	ret = listen(s, SOMAXCONN);
	if(SOCKET_ERROR == ret)
	{
		printf("Listen Error...\n");
		return -1;
	}

	int len = 0;
	while(true)
	{
		S_PARA* pPara = new S_PARA;
		len = sizeof(pPara->caddr);

		pPara->cs = accept(s, (SOCKADDR*)&pPara->caddr, &len);
		if(INVALID_SOCKET == pPara->cs)
		{
			printf("Accept Error...\n");
			break;
		}
		else
		{
			printf("Create New User...\n");
		}

		if(NULL != ServerProc)
		{
			CreateThread(NULL, 0, ServerThread, pPara, 0, NULL);
		}
		else
		{
			delete pPara;
		}
	}

	WSACleanup();

	return 0;
}

//---------------------------------------------------------------------

int IniTCPClient(char* ip, int port, int(*fn)(SOCKET s))
{
	int ret = 0;
	WSADATA wsa;

	ret = WSAStartup(MAKEWORD(2, 2), &wsa);
	if(0 != ret)
	{
		printf("WinSock Error...\n");
		return -1;
	}

	SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
	if(INVALID_SOCKET == s)
	{
		printf("Socket Error...\n");
		return -1;
	}

	SOCKADDR_IN saddr;
	ZeroMemory(&saddr, sizeof(saddr));

	saddr.sin_family = AF_INET;
	saddr.sin_addr.s_addr = inet_addr(ip);
	saddr.sin_port = htons(port);

	ret = connect(s, (SOCKADDR*)&saddr, sizeof(saddr));
	if(SOCKET_ERROR == ret)
	{
		printf("Connect Error...\n");
	}
	else
	{
		printf("Connect Success...");
	}

	if(NULL != fn)
	{
		fn(s);
	}

	closesocket(s);

	WSACleanup();

	return 0;
}

//---------------------------------------------------------------------