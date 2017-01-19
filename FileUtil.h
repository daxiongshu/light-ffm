#pragma once
#ifndef _FILEUTIL_H
#define _FILEUTIL_H
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "StringUtil.h"
#include <cstring>
#include <iostream>
#define MAXLINESIZE 100000
class FileUtil
{
public:
	FileUtil();
	~FileUtil();
        // read csv-like file where num of cols is fixed
        // X, Y, ID, length are return values;
        // columns is optional return value;
        template<typename T1, typename T2>  static void read_csv(const std::string &fileName,
                const std::string &delimiter, T1* &X, int & length,
                T2* &Y, int* &ID,
                std::string idName = "-1", std::string labelName = "-1",
                bool verbose=0);

	template<typename T1, typename T2> static void writeSubmission(T1 *ID, T2 *yp, int num, 
		std::string fileName, std::string idName, std::string predName);

	
	static int countCols(const std::string &fileName, const std::string &delimiter,
		std::string idName = "-1", std::string labelName = "-1", 
		int* idCol=nullptr, int*labelCol=nullptr);
	// doesn't include id column and label column if not -1

	static void split_sample(const std::string &train, 
		const std::string &tr, const std::string &va, int portion=10);
	// be default, take 1/10 as tr data and 1/20 as va data, in round robin way

	static long long countRows(const std::string &fileName);
	static bool fileExist(const std::string &fileName);
	template<typename T> static void writeNumber(FILE *fp, T x);
	template<typename T> static T readNumber(FILE *fp);
	static void sanityCheck(const std::string &fileName,std::string idName, std::string labelName);
private:
	//static std::vector<std::string> columns_;
};


template<typename T1, typename T2> void FileUtil::writeSubmission(T1 *ID, T2 *yp, int num,
		std::string fileName, std::string idName, std::string predName){

	printf("writing submission\n");
    	std::ofstream myfile;
    	myfile.open (fileName);
        myfile<<idName<<","<<predName<<std::endl;
    	for (int i=0;i<num;i++){
            	myfile<<ID[i]<<","<<yp[i]<<std::endl;
    	}
    	myfile.close();	
}



template<typename T> void FileUtil::writeNumber(FILE *fp, T x){

	if (fp != NULL) {
		T xBuf[10];
		xBuf[0] = x;
		fwrite(xBuf,sizeof(T),1,fp);
	}
}

template<typename T> T FileUtil::readNumber(FILE *fp){
	if (fp != NULL) {
		T xBuf[10];
		fread(xBuf, sizeof(T), 1, fp);
		return xBuf[0];
	}
	return 0;
}

template<typename T1, typename T2> void FileUtil::read_csv(const std::string &fileName,
                const std::string &delimiter, T1* &X, int & length,
                T2* &Y, int* &ID, std::string idName, std::string labelName,
                bool verbose)
{

	if (verbose)
	{
		printf("readCSV %s\n", fileName.c_str());
	}
	
	std::clock_t startTime = std::clock();
	
	int idCol = -1;
	int labelCol = -1;
	int num_cols = countCols(fileName,delimiter,idName,labelName,
		&idCol,&labelCol);
	int num_rows = (int)countRows(fileName);

	if (verbose)
	{
		printf("rows: %d cols: %d, idCol: %d labelCol: %d\n",
			num_rows, num_cols, idCol, labelCol);
	}

	length = (num_rows-1) * num_cols;
	X = new T1[length];
	ID = nullptr;
	Y = nullptr;

	if (idCol!=-1)
	{
		ID = new int[num_rows-1];
	}
	if (labelCol!=-1)
	{
		Y = new T2[num_rows-1];
	}

	FILE *f = fopen(fileName.c_str(), "r");
	
	char line[MAXLINESIZE];
	fgets(line, MAXLINESIZE, f); // ignore the header
	int p = 0;
	for (int i=0; fgets(line, MAXLINESIZE, f) != nullptr;i++)
	{
		if (verbose && i%100000 == 0)
		{
			double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
			printf("Time: %fs Lines: %d\n", duration, i);
		}
		int tmp = 0;
		char* pch = strtok(line, delimiter.c_str());
		while (pch!=nullptr)
		{
			if (tmp==labelCol)
			{
				Y[i] = (T2) atof(pch);
			}
			else if (tmp==idCol)
			{
				ID[i] = atoi(pch);
			}
			else
			{
				X[p++] = (T1) atof(pch);
			}
			tmp++;
			pch = strtok(NULL,delimiter.c_str());
		}
	}
	double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
	printf("readCSV process time: %f, rows: %d, columns: %d\n", duration, num_rows, num_cols);
	if (verbose)
	{	
		printf("last two lines\n");

		// total linex expect header is num_rows - 1
		for (size_t i = num_rows-3; i < num_rows-1; i++)
		{
			if (idCol != -1){
				//printf("%s: %d ",idName.c_str(),ID[i]);
                                std::cout<<idName<<": "<<ID[i]<<" ";
			}
			printf("X: ");
			for (size_t j = 0; j < num_cols; j++)
			{
				//printf("%f ", X[i*num_cols + j]);
				std::cout<<X[i*num_cols + j]<<" ";
			}
			if (labelCol!=-1)
			{
				//printf("%s: %f",labelName.c_str(),Y[i]);
				std::cout<<labelName<<": "<<Y[i];
			}
			printf("\n");
		}
	}
}
#endif
