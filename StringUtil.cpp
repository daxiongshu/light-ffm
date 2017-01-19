#include "StringUtil.h"
#include <cstring>
std::string StringUtil::join(std::string*A, const std::string &delimiter,
	int length, bool strip_){
	if (length == 0)
	{
		return "";
	}
	if (length == 1)
	{
		return A[0];
	}
	std::string result = strip_ ? strip(A[0]) : A[0];
	for (size_t i = 1; i < length; i++)
	{
		result = strip_ ? result + delimiter + strip(A[i]) : result + delimiter + A[i];
	}
	return result;


}

std::string StringUtil::join(const std::vector<std::string> &A,
	const std::string &delimiter, bool strip_){
	if (A.size()==0)
	{
		return "";
	}
	if (A.size()==1)
	{
		return A[0];
	}
	std::string result = strip_ ? strip(A[0]) : A[0];
	for (size_t i = 1; i < A.size(); i++)
	{
		result = strip_ ? result + delimiter + strip(A[i]) : result + delimiter + A[i];
	}
	return result;
}

std::string StringUtil::strip(const std::string s){
	const std::string& whitespace = " \t\n\v\f\r";
	const auto strBegin = s.find_first_not_of(whitespace);
	if (strBegin==std::string::npos)
	{
		return "";
	}
	const auto strEnd = s.find_last_not_of(whitespace);
	const auto strRange = strEnd - strBegin + 1;
	return s.substr(strBegin, strRange);
}

void StringUtil::stripInplace(std::string &s){

	const std::string& whitespace = " \t\n\v\f\r";
	const auto strBegin = s.find_first_not_of(whitespace);
	if (strBegin == std::string::npos)
	{
		s="";
	}
	const auto strEnd = s.find_last_not_of(whitespace);
	const auto strRange = strEnd - strBegin + 1;
	s = s.substr(strBegin, strRange);
}

std::vector<std::string> StringUtil::split(const std::string &s, 
	const std::string &delimiter, bool strip_){

	std::vector<std::string> result;
	size_t start = 0;
	size_t range = s.length();
	size_t end = 0;

	while ((end = s.substr(start,range).find(delimiter)) != std::string::npos)
	{
		range = end;
		if (strip_)
		{
			result.push_back(strip(s.substr(start,range)));
		}
		else
		{
			result.push_back(s.substr(start, range));
		}
		start = start + range + 1;
		if (start>=s.length())
		{
			result.push_back("");
			return result;
		}
		range = s.length() - start + 1;
	}
	if (strip_)
	{
		result.push_back(strip(s.substr(start, range)));
	}
	else
	{
		result.push_back(s.substr(start, range));
	}
	return result;
}

std::string StringUtil::replace(const std::string &s,
	const std::string &search, const std::string &replace){

	std::string result = s;
	for (size_t pos = 0;; pos += replace.length())
	{
		pos = result.find(search, pos);
		if (pos == std::string::npos)
		{
			break;
		}
		result.erase(pos, search.length());
		result.insert(pos, replace);
		
	}
	return result;
}

int StringUtil::replaceInplace(std::string &s,
	const std::string &search, const std::string &replace){
	// find pattern &search in s, and replace it with &replace
	// replace all such patterns in place
	int count = 0; // number of patterns replaced;
	for (size_t pos = 0; ; pos+=replace.length())
	{
		pos = s.find(search, pos);
		if (pos==std::string::npos)
		{
			break;
		}
		s.erase(pos, search.length());
		s.insert(pos, replace);
		count++; 
	}
	return count;
}


bool StringUtil::startsWith(const std::string &s, const std::string &pattern){

	size_t ls = s.length();
	size_t lp = pattern.length();
	if (ls<lp)
	{
		return false;
	}
	int result = s.substr(0, lp).compare(pattern);
	return result == 0;
}

bool StringUtil::endsWith(const std::string &s, const std::string &pattern){

	size_t ls = s.length();
	size_t lp = pattern.length();
	if (ls<lp)
	{
		return false;
	}
	int result = s.substr(ls-lp, lp).compare(pattern);
	return result == 0;
}

void StringUtil::sanityCheck(){

	// starts with and end with
	std::string s= "here is a test";
	std::string pattern = "here";
	printf("'%s' starts with '%s': %d\n", s.c_str(), 
		pattern.c_str(), startsWith(s, pattern));
	printf("'%s' ends with '%s': %d\n", s.c_str(),
		"test", endsWith(s, "test"));

	// replace
	pattern = "is";
	int result = replaceInplace(s, pattern, "are");
	if (result)
		printf("%s\n", s.c_str());
	size_t t = 10000000000;
	printf("max of size_t: %lu\n", t);

	// split
	std::vector<std::string> results = split(s," ");
	for (size_t i = 0; i < results.size(); i++)
	{
		printf("%s\n", results[i].c_str());
	}
	printf("join: %s\n", join(results, ",").c_str());
}

StringUtil::StringUtil()
{
}


StringUtil::~StringUtil()
{
}
