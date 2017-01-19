#pragma once
#ifndef _SORT_H
#define _SORT_H
#include "Utility.h"
#include <cstdio>
#include <cstdlib>
//using namespace std;
template <typename T>class Sort
{
public:
	Sort();
	
	static void bubbleSort(T *A, int length);

        static void quickSortPair(T *A, T* B, int length, bool ascending);
        static int partitionPair(T *A, T* B, int length, bool ascending);



	static int* quickSortIndex(T *A, int length); // sort index, instead of instance T
        static void quickSortIndex(T *A, int* indexArray, int length);
	static int partitionIndex(T *A, int* indexArray, int length);

	static void quickSort(T *A, int length);
	static int partition(T *A, int length);

	static void mergeSort(T *A, int length);
	static void insertSort(T *A, int length);
	static void countSort(T *A, int length);
	static void heapSort(T *A, int length);
	static void buildHeap(T *A, int length); // O(n)
	static void maxHeapify(T *A, int length, int id);


	static void quickSort(std::vector<T> &A);
	static void quickSort(std::vector<T> &A, int start, int end);
	static int partition(std::vector<T> &A, int stat, int end);
	static T quickSelect(std::vector<T> &A, int start, int end, int k); // select kth smallest element
	static T quickSelect(std::vector<T> &A, int k); // select kth smallest element

	static void maxHeapify(std::vector<T> &A, int start, int end);
	static void buildHeap(std::vector<T> &A);
	static void heapSort(std::vector<T> &A);

	~Sort();
private:
};

template<typename T> void Sort<T>::quickSortPair(T *A, T* B, int length, bool ascending){
        //printf("length:%d\n",length);
        if (length <= 1) return;
        int pivot_pos = partitionPair(A, B, length, ascending);
        quickSortPair(A, B, pivot_pos, ascending);
        quickSortPair(A + pivot_pos + 1, B + pivot_pos + 1, length - pivot_pos - 1, ascending);


}

template<typename T>int Sort<T>::partitionPair(T *A, T* B, int length, bool ascending){
        int pivot = rand() % length;
        Utility<T>::swap(A, length-1, pivot);
        Utility<T>::swap(B, length-1, pivot);
        int r = length - 1;
        int p = 0;
        int i = p - 1;
        int j = p;
        for (size_t k = p; k < r; k++)
        {
                if (((ascending==1)&&(A[j] < A[r])) || ((ascending==0)&&(A[j] > A[r]))){
                        i += 1;
                        Utility<T>::swap(A, i, j);
			Utility<T>::swap(B, i, j);
                }
                j++;
        }
        Utility<T>::swap(A, i + 1, r);
        Utility<T>::swap(B, i + 1, r);
        return i + 1;

}


template<typename T> int* Sort<T>::quickSortIndex(T *A, int length){
	int* result = new int[length];

	for(int i=0;i<length;i++) 
		result[i] = i;

        quickSortIndex(A,result,length);
        return result;
}


template<typename T> void Sort<T>::quickSortIndex(T *A, int* indexArray, int length){

	if (length <= 1) return;
        int pivot_pos = partitionIndex(A, indexArray, length);
        quickSortIndex(A, indexArray, pivot_pos);
        quickSortIndex(A, indexArray + pivot_pos + 1, length - pivot_pos - 1);


}

template<typename T>int Sort<T>::partitionIndex(T *A, int* indexArray, int length){
        int pivot = rand() % length;
        Utility<int>::swap(indexArray, length-1, pivot);
        int r = length - 1;
        int p = 0;
        int i = p - 1;
        int j = p;
        for (size_t k = p; k < r; k++)
        {
                if (A[indexArray[j]] < A[indexArray[r]]){
                        i += 1;
                        Utility<int>::swap(indexArray, i, j);
                }
                j++;
        }
        Utility<int>::swap(indexArray, i + 1, r);
        return i + 1;

}


template <typename T> T Sort<T>::quickSelect(std::vector<T> &A, int start, int end, int k){
	if (end - start<1) return A[start];
	int pivot_pos = partition(A, start, end);
	if (pivot_pos == k - 1){
		
		return A[pivot_pos];
	}
	else
	{
		//printf("pivot %d\n", pivot_pos);
		if (pivot_pos>k-1)
		{
			return quickSelect(A, start, pivot_pos-1, k);
		}
		else
		{
			return quickSelect(A, pivot_pos+1, end, k);
		}
	}
}

template <typename T> T Sort<T>::quickSelect(std::vector<T> &A, int k){
	return quickSelect(A, 0, A.size() - 1, k);
}

template<typename T>int Sort<T>::partition(std::vector<T> &A, int start, int end){
	int pivot = rand() % (end - start) + start;
	Utility<T>::swap(A, end, pivot);
	int i = start - 1;
	int j = start;
	for (int k = start; k < end; k++)
	{
		if (A[k]<A[end]){
			i++; 
			Utility<T>::swap(A, i, k);
		}
		j++;
	}
	Utility<T>::swap(A, i+1, end);
	return i + 1;
}

template <typename T> void Sort<T>::quickSort(std::vector<T> &A){
	quickSort(A, 0, A.size() - 1);
}

template <typename T> void Sort<T>::quickSort(std::vector<T> &A, int start, int end){
	if (end-start<1) return;
	int pivot_pos = partition(A,start,end);
	quickSort(A, start, pivot_pos-1);
	quickSort(A, pivot_pos + 1, end);

}

template <typename T> void Sort<T>::bubbleSort(T *A, int length){
	for (size_t i = 0; i < length - 1; i++){
		for (size_t j = length - 1; j>i; j--)
		{
			if (A[j]<A[i])
			{
				Utility<T>::swap(A, i, j);
			}
		}
	}
}
template<typename T> void Sort<T>::maxHeapify(std::vector<T> &A, int start, int end){
	int sudo_index = start + 1;
	while (sudo_index * 2 <= end+1){
		int left = sudo_index * 2;
		int right = sudo_index * 2 + 1;
		int largest = sudo_index;
		// left-1 and right-1 are the real index
		if (A[largest - 1] < A[left - 1]){
			largest = left;
		}

		if (right<=end + 1 && A[largest - 1]<A[right - 1])
		{
			largest = right;
		}

		if (largest != sudo_index)
		{
			Utility<T>::swap(A, sudo_index - 1, largest - 1);
			sudo_index = largest;
		}
		else
		{
			break;
		}
		//Utility<T>::printArray(A,  "heapify");
		//printf("largest %d sudo index %d length %d\n", largest, sudo_index,length);

	}
}
template<typename T> void Sort<T>::maxHeapify(T *A, int length, int id){
	// index translation
	int sudo_index = id + 1;
	while (sudo_index * 2 < length + 1){
		int left = sudo_index * 2;
		int right = sudo_index * 2 + 1;
		int largest = sudo_index;
		// left-1 and right-1 are the real index
		if (A[largest - 1] < A[left - 1]){
			largest = left;
		}

		if (right<length + 1 && A[largest - 1]<A[right - 1])
		{
			largest = right;
		}

		if (largest != sudo_index)
		{
			Utility<T>::swap(A, sudo_index - 1, largest - 1);
			sudo_index = largest;
		}
		else
		{
			break;
		}
		//Utility::printArray(A, length, "heapify");
		//printf("largest %d sudo index %d length %d\n", largest, sudo_index,length);

	}

}

template<typename T>void Sort<T>::buildHeap(std::vector<T> &A){
	for (int i = A.size() / 2; i >= 0; i--)
	{
		//printf("call max heapify i=%d\n",i);
		maxHeapify(A, i, A.size()-1);
	}
	//printf("heap is built\n");
}

template<typename T>void Sort<T>::buildHeap(T *A, int length){
	for (int i = length / 2; i >= 0; i--)
	{
		//printf("call max heapify i=%d\n",i);
		maxHeapify(A, length, i);
	}
	//printf("heap is built\n");
}
template<typename T>void Sort<T>::heapSort(T *A, int length){
	buildHeap(A, length);
	for (int i = length - 1; i > 0; i--)
	{
		Utility<T>::swap(A, 0, i);
		maxHeapify(A, i, 0);
	}
}

template<typename T>void Sort<T>::heapSort(std::vector<T> &A){
	buildHeap(A);
	for (int i = A.size() - 1; i > 0; i--)
	{
		Utility<T>::swap(A, 0, i);
		maxHeapify(A, 0, i-1 );
	}
}

template<typename T> void Sort<T>::quickSort(T *A, int length){
	if (length <= 1) return;
	int pivot_pos = partition(A, length);
	quickSort(A, pivot_pos);
	quickSort(A + pivot_pos + 1, length - pivot_pos - 1);

}

template<typename T>int Sort<T>::partition(T *A, int length){
	int pivot = rand() % length;
	Utility<T>::swap(A, length-1, pivot);
	int r = length - 1;
	int p = 0;
	int i = p - 1;
	int j = p;
	for (size_t k = p; k < r; k++)
	{
		if (A[j] < A[r]){
			i += 1;
			Utility<T>::swap(A, i, j);
		}
		j++;
	}
	Utility<T>::swap(A, i + 1, r);
	return i + 1;

}

#endif
