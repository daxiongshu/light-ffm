#include "FileUtil.h"

int FileUtil::countCols(const std::string &fileName, const std::string &delimiter,
	std::string idName, std::string labelName,
	int* idCol, int*labelCol){

	if (fileExist(fileName) == 0)
	{
		return -1;
	}	
	int n_columns = 0;
	FILE *f = fopen(fileName.c_str(), "r");
	
	char line[MAXLINESIZE];
	fgets(line, MAXLINESIZE, f);
	char* pch = strtok(line, delimiter.c_str());
	int count = 0;
	while (pch!=nullptr)
	{
		printf("%s\n", pch);
		std::string tmp = StringUtil::strip(std::string(pch));
		if (idName == tmp){
			if (idCol!=nullptr)
			{
				*idCol = count;
			}
		}
		else if (labelName == tmp)
		{
			if (labelCol!=nullptr)
			{
				*labelCol = count;
			}
		}else
		{
			n_columns++;
		}		
		pch = strtok(NULL, delimiter.c_str());
		count++;
	}
	fclose(f);
	
	delete[] pch;
	return n_columns;
}

long long FileUtil::countRows(const std::string &fileName){

	if (fileExist(fileName)==0)
	{
		return -1;
	}
	long long totalLines=0;
	FILE *f = fopen(fileName.c_str(), "r");
	
	char line[MAXLINESIZE];
	for (; fgets(line, MAXLINESIZE, f) != nullptr;totalLines+=1)
	{}
	fclose(f);
	return totalLines;
}


bool FileUtil::fileExist(const std::string &fileName){

	std::ifstream f(fileName);
	bool result = f.good();
	f.close();
	return result;

}

void FileUtil::sanityCheck(const std::string &fileName, std::string idName, std::string labelName){
	bool exist = fileExist(fileName);
	printf("%s exist? %d\n", fileName.c_str(), exist);
	//printf("%s lines: %d\n", fileName.c_str(), countRows(fileName));

	std::string tr = StringUtil::replace(fileName, "train.csv", "tr.csv");
	std::string va = StringUtil::replace(fileName, "train.csv", "va.csv");

	if (fileExist(va) == false)
	{
		split_sample(fileName, tr, va);
	}

	float *X = nullptr;
	long long *Y = nullptr;
	int* ID = nullptr;
	int length = 0;
	
	std::vector<std::string> columns;
        read_csv<float, long long>(va, ",", X, length,
                Y, ID, idName, labelName,
                true);
}

void FileUtil::split_sample(const std::string &train,
	const std::string &tr, const std::string &va, int portion){

	FILE *f = fopen(train.c_str(), "r");
	FILE *ftr = fopen(tr.c_str(), "w");
	FILE *fva = fopen(va.c_str(), "w");

	char line[MAXLINESIZE];
	fgets(line, MAXLINESIZE, f); // get the header
	fputs(line, ftr);
	fputs(line, fva);
	
	for (int i = 0; fgets(line, MAXLINESIZE, f) != nullptr; i++)
	{
		if (i % portion == rand() % portion)
		{
			fputs(line, ftr);
		}
		else if (i % (portion*2) == rand() % (portion*2))
		{
			fputs(line, fva);
		}
	}
	fclose(f);
	fclose(ftr);
	fclose(fva);
}

FileUtil::FileUtil()
{
}


FileUtil::~FileUtil()
{
}
