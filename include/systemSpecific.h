/*
 * systemSpecific.h
 *
 *  Created on: Mar 13, 2019
 *      Author: tumaak
 */

#ifndef SYSTEMSPECIFIC_H_
#define SYSTEMSPECIFIC_H_
#include <string>
#ifdef WIN_OS
	#include <Windows.h>
    #include "stdafx.h"
    #include <cstdlib>
#endif
#ifdef LINUX_OS
	#include <unistd.h>
#endif

namespace mySys {
#ifdef LINUX_OS
	typedef pid_t myPidType;

	myPidType getPid(){
		return getpid();
	}
	void mkdir(std::string fileName){
		system(("mkdir -p " + fileName).c_str());

	}
#endif // LINUX_OS

#ifdef WIN_OS
	typedef DWORD myPidType;

	myPidType getPid(){
		return GetCurrentProcessId();
	}

    void mkdir(std::string fileName) {
        //CreateDirectory(fileName.c_str(), NULL);
        std::system(("mkdir "+fileName).c_str());
	}
#endif // WIN_OS


}




#endif /* SYSTEMSPECIFIC_H_ */
