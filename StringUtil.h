#pragma once
#ifndef _STRINGUTIL_H
#define _STRINGUTIL_H
#include <string>
#include <vector>
class StringUtil
{
public:
	StringUtil();
	~StringUtil();
	static int replaceInplace(std::string &s, 
		const std::string &search, const std::string &replace);
	static std::string replace(const std::string &s,
		const std::string &search, const std::string &replace);

	static std::string join(std::string*A, const std::string &delimiter, 
		int length, bool strip_ = false);

	static std::string join(const std::vector<std::string> &A, 
		const std::string &delimiter, bool strip_ =false);

	static std::string strip(const std::string s);
	static void stripInplace(std::string &s);

	static std::vector<std::string> split(const std::string &s, 
		const std::string &delimiter, bool strip_=false);

	static bool startsWith(const std::string &s, const std::string &pattern);
	static bool endsWith(const std::string &s, const std::string &pattern);
	static void sanityCheck();
};

#endif