#pragma once
#ifndef _UTILITY_H
#define _UTILITY_H
#include <vector>
#include <cstdlib>
#include <iostream>
#include <string>

// assumption is integer, float or double type
template <typename T>class Utility
{
public:
	Utility();
	static T hash(const std::string &s); // hash string to T, int or float
	static std::string toNBase(T a, const int & base); // map a dec to n-based number
	static void swap(T *A, int i, int j);
	static void swap(std::vector<T> &A, int i, int j);
	static T* genRandomArray(int length);
	static void genRandomArray(std::vector<T> &A, int length);
	static void printArray(T *A, int length, std::string name);
	static void printArrayIndex(T *A, int* indexArray, int length, std::string name);
	static void printArray(std::vector<T> &A, std::string name);
	~Utility();
};

template <typename T> T Utility<T>::hash(const std::string &s){

	int hash_ = 0;
	int offset = 'a' - 1;
	for(std::string::const_iterator it=s.begin(); it!=s.end(); ++it) {
  		hash_ = hash_ << 1 | (*it - offset);
	}
        return (T) hash_;
}

template <typename T> std::string Utility<T>::toNBase(T a, const int & base){
	if (a/base==0)
	{
		return std::to_string(a);
	}
	std::string result = toNBase(a / base, base);
	return result + std::to_string(a%base);

}

template <typename T> void Utility<T>::swap(std::vector<T> &A, int i, int j){
	T tmp;
	tmp = A[i];
	A[i] = A[j];
	A[j] = tmp;
}

template <typename T> void Utility<T>::swap(T *A, int i, int j){
	T tmp = A[i];
	A[i] = A[j];
	A[j] = tmp;
}
template <typename T> void Utility<T>::genRandomArray(std::vector<T> &A, int length){
	
	for (size_t i = 0; i < length; i++)
	{
		A.push_back(rand()%100);
	}
	
}
template <typename T> T* Utility<T>::genRandomArray(int length){
	int *A = (int*)malloc(length*sizeof(int));
	for (size_t i = 0; i < length; i++)
	{
		A[i] = rand()%100;
	}
	return A;
}
template <typename T> void Utility<T>::printArray(std::vector<T> &A, std::string name){
	std::cout << "print Array " << name << std::endl;
	for (size_t i = 0; i < A.size(); i++)
	{
		std::cout << A[i] << " ";
		if (i>10)
		{
			break;
		}
	}
	std::cout << std::endl;
}
template <typename T> void Utility<T>::printArray(T *A, int length, std::string name){
	std::cout << "print Array " << name << std::endl;
	for (size_t i = 0; i < length; i++)
	{
		std::cout << A[i] << " ";
		if (i>10)
		{
			break;
		}
	}
	std::cout << std::endl;
}

template <typename T> void Utility<T>::printArrayIndex(T *A, int* indexArray, int length, std::string name){
        std::cout << "print Array " << name << std::endl;
        for (size_t i = 0; i < length; i++)
        {
                std::cout << A[indexArray[i]] << " ";
                if (i>10)
                {
                        break;
                }
        }
        std::cout << std::endl;
}

#endif 
