/*
 * fileWrapper.h
 *
 *  Created on: Mar 13, 2019
 *      Author: tumaak
 */

#ifndef FILEWRAPPER_H_
#define FILEWRAPPER_H_

#include <string>
#include <cstdio>

class fileWrapper {
	FILE* f;
public:
	fileWrapper(std::string fileName, std::string flags){
		f = std::fopen(fileName.c_str(), flags.c_str());
	}

	void rewind(){
		if(f!=NULL){
			std::rewind(f);
		}
	}

	//ŧemplate <typename T>
	void write(const void* data, int size){
		std::fwrite(data,size,1,f);
	}

	void flush(){
		std::fflush(f);
	}

	void writeAndFlush(const void* data, int size){
		std::fwrite(data,size,1,f);
		std::fflush(f);
	}

	void read(void* data, int size){
		std::fread(data, size, 1, f);
	}

	void close(){
		if(f!=NULL){
			std::fclose(f);
		}
	}

	bool is_opened(){
		return f!=NULL;
	}

	~fileWrapper(){
		if(f!=NULL){
			std::fclose(f);
		}
	}
};

#endif /* FILEWRAPPER_H_ */
