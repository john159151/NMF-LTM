#include "stdio.h"
#include <iostream>
#include <string>
#include "data.cpp"
#include "matrix.cpp"
#include "LMLTM_Update.cpp"
#include "util.cpp"
#include <omp.h>
#include "RecordInfo.h"

using namespace std;

#pragma once
class Lifelong_Machine_Learning_Topic_Model
{
public:
	//constructors
	Lifelong_Machine_Learning_Topic_Model()
	{
	}

	~Lifelong_Machine_Learning_Topic_Model() {}

	dMatrix running(const dMatrix &D, const dMatrix &UInit, const dMatrix &VInit, int K, int dNum, string folderPath, double alpha, double beta, double lamda, double gamma, double eta, int outsideIterNum, int subIterNum, int topK, string outputFolder, string wordmapFilePath, string subWordmapFilePath, string subLabelFilePath, int recordSMatrixWay)
	{
		dataIn d;
		int N = D.colno();
		int M = D.rowno();

		//使用固定的初始化U, V矩阵
		dMatrix U = UInit;
		dMatrix V = VInit;

		//generate subS and subA matrix
		dMatrix subS(M, M), subA(M, M);
		if (recordSMatrixWay == 3)
		{
			RecordInfo::getSubSMatrixWithSmooth(wordmapFilePath, subWordmapFilePath, subS);
		}
		else
		{
			RecordInfo::getSubSMatrix(wordmapFilePath, subWordmapFilePath, subS);
		}
		RecordInfo::getSubAMatrix(subS, subA);

		//generate subW and subT matrix
		dMatrix subW(N, N), subT(N, N);
		RecordInfo::generateWMatrix(subLabelFilePath);
		RecordInfo::getSubAMatrix(subW, subT);

		string s1 = to_string(K);
		string suffixStr = to_string(subIterNum);
		
		LMLTM_Update r;
		for (int i = 0; i<outsideIterNum; i++)
		{
			char s[100];
			sprintf(s, "%d", i + 1);
			r.updateU(D, U, V, alpha, beta, eta, subS, subA);
			r.updateV(D, U, V, lamda, gamma, eta, subW, subT);
			if ((i + 1) % outsideIterNum == 0)
			{
				d.save(U, outputFolder + folderPathSplit + "U" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
				d.save(V, outputFolder + folderPathSplit + "V" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
			}
			cout << "K=" << K << " iter:" << i << endl;
		}
		//save vector info
		//d.save(RecordInfo::updateUIterNum, outputFolder + folderPathSplit + "updateUIterNum" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
		//d.save(RecordInfo::diffInEachUpdateUIter, outputFolder + folderPathSplit + "diffInEachUpdateUIter" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
		//d.save(RecordInfo::updateUAllTime, outputFolder + folderPathSplit + "updateUAllTime" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
		//d.save(RecordInfo::updateUEachIterTime, outputFolder + folderPathSplit + "updateUEachIterTime" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
		//d.save(RecordInfo::orthogonalityInEachUpdateUIter, outputFolder + folderPathSplit + "orthogonalityInEachUpdateUIter" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
		//d.save(RecordInfo::uSparsenessInEachUpdateUIter, outputFolder + folderPathSplit + "uSparsenessInEachUpdateUIter" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
		//d.save(RecordInfo::vSparsenessInEachUpdateVIter, outputFolder + folderPathSplit + "vSparsenessInEachUpdateVIter" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
		//d.save(RecordInfo::diffBetweenDAndUVInEachUpdateU, outputFolder + folderPathSplit + "diffBetweenDAndUVInEachUpdateU" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");
		//d.save(RecordInfo::trUTLUInEachUpdateU, outputFolder + folderPathSplit + "trUTLUInEachUpdateU" + s1 + "_" + to_string(dNum) + "_" + suffixStr + ".txt");

		return U;
	}

private:
	void checkMatrixZero(const dMatrix &X)
	{
		for (int i = 0; i < X.rowno(); i++)
		{
			for (int j = 0; j < X.colno(); j++)
			{
				if (X(i, j) != 0)
				{
					cout << "not zero!!! " << X(i, j) << endl;
					return;
				}
			}
		}
		cout << "zero" << endl;
	}

	void adjustMinValueOfMatrix(dMatrix &X, double minValueSet)
	{
		for (int i = 0; i < X.rowno(); i++)
		{
			for (int j = 0; j < X.colno(); j++)
			{
				if (X(i, j) < minValueSet)
				{
					X(i, j) = minValueSet;
				}
			}
		}
	}

	double eps = 1e-9;

#ifdef Linux
	const string folderPathSplit = "/";
#else
	const string folderPathSplit = "\\";
#endif
};

