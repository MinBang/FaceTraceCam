#include <WinSock2.h>

#pragma comment(lib, "ws2_32")

int recvn(SOCKET s, char* buf, int len, int flags);

int IniTCPServerM(int port, int(*Serverproc)(SOCKET c));

int IniTCPClient(char* ip, int port, int(*fn)(SOCKET s));