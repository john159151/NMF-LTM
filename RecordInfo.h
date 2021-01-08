#ifndef RECORDINFO_H
#define RECORDINFO_H

#include "stdio.h"
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include "matrix.cpp"
#include <math.h>
#include "SMatrixNode.cpp"

using namespace std;

typedef techsoft::matrix<double> dMatrix;

#pragma once
class RecordInfo
{
public:
	//constructors
	RecordInfo();

	~RecordInfo();

	static vector<int> updateUIterNum;
	static vector<double> diffInEachUpdateUIter;
	static vector<double> eachDocAllTime;
	static vector<double> updateUAllTime;
	static vector<double> updateUEachIterTime;
	static vector<double> orthogonalityInEachUpdateUIter;
	static vector<double> uSparsenessInEachUpdateUIter;
	static vector<double> vSparsenessInEachUpdateVIter;
	static vector<double> diffBetweenUTDAndVInEachUpdateUIter; //U^TD-V的F范数的平方
	static vector<double> diffBetweenUTUAndIInEachUpdateUIter; //U^TU-I的F范数的平方
	static vector<double> diffBetweenDAndUVInEachUpdateU; //D-UV的F范数的平方
	static vector<double> trUTLUInEachUpdateU; //tr(UTLU)

	//globe tyep, no need to reset
	// SMatrix and AMatrix corresponds to the affinitity between any two words in the priors
	static vector<SMatrixNode> *SMatrix;
	static int SMatrixSize;
	//static dMatrix AMatrix;

	// WMatrix and TMatrix corresponds to the affinitity between any two samples(documents) supervised by class labels.
	//static dMatrix WMatrix;
	//static dMatrix TMatrix;

	static double getSMatrixValue(int rowIndex, int columnIndex); //index start from 0

	static void setSMatrixValue(int rowIndex, int columnIndex, double value); //index start from 0

	static dMatrix generateWMatrix(string labelFilePath);

	static void setMatrixZero(dMatrix &X);

	static void setMatrixZero(double **X, int rowNum, int columnNum);

	static void setMatrixZero(vector<SMatrixNode>* X, int rowNum);

	static void getSubAMatrix(const dMatrix &subS, dMatrix &subA);

	static void getSubSMatrix(string wordmapFilePath, string subWordmapFilePath, dMatrix &subS);

	static void getSubSMatrixWithSmooth(string wordmapFilePath, string subWordmapFilePath, dMatrix &subS);

	static void setTrUTLUInEachUpdateU(const dMatrix &U, const dMatrix &A, const dMatrix &S);

	static void setDiffBetweenDAndUVInEachUpdateU(const dMatrix &U, const dMatrix &D, const dMatrix &V);

	static void setDiffBetweenUTUAndIInEachUpdateUIter(const dMatrix &U);

	static void setDiffBetweenUTDAndVInEachUpdateUIter(const dMatrix &U, const dMatrix &D, const dMatrix &V);

	static void setVSparsenessInEachUpdateVIter(const dMatrix &V);

	static void setUSparsenessInEachUpdateUIter(const dMatrix &U);

	static void setOrthogonalityInEachUpdateUIter(const dMatrix &U);

	static void setDiffInEachUpdateUIter(double diff);

	static void setUpdateUIterNum(int num);

	static void timeRecordStart(bool allOrNot);

	static void timeRecordEnd(bool allOrNot);

	static void timeRecordStart();

	static void timeRecordEnd();

	static dMatrix mul(const dMatrix &A, const dMatrix &B);

	static void clearAllWithoutGlobalType();

	static void clearAllGlobalType();

	static double eps;

private:
	static time_t tStart, tEnd;
	static time_t tStartAll, tEndAll;
	static time_t tStartAllEachDoc, tEndAllEachDoc;
	static double calSparsenessOfMatrix(const dMatrix &X);
};

#endif