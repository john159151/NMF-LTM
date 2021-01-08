#include "stdio.h"
#include <iostream>
#include <string>
#include "data.cpp"
#include "matrix.cpp"
#include "LMLTM_Update.cpp"
#include "util.cpp"
#include <omp.h>
#include "RecordInfo.h"
#include "Lifelong_Machine_Learning_Topic_Model.cpp"
#include "TopicWords.cpp"

#ifdef Linux
#include <sys/types.h>  
#include <sys/stat.h>
#else
#include <windows.h> 
#endif // Linux

using namespace std;

#pragma once
class LMLTM_Framework
{
public:
	//constructors
	LMLTM_Framework()
	{
	}

	~LMLTM_Framework() {}

	void running(double alpha, double beta, double lamda, double gamma, double eta, string folderPath, string folderPathSplit, 
		int iterNum, int iterNumBP, int topK, int knowledgeGainLimit, int DocNum, string wordmapFilePath, 
		const vector<dMatrix> &UInit, const vector<dMatrix> &VInit, int recordSMatrixWay, int K, 
		const vector<string> &subTfidfFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subLabelFilePath, 
		const int splitDatasetWay, const bool needToUseNewSMatrix)
	{
		if (needToUseNewSMatrix == true)
		{
			RecordInfo::clearAllGlobalType();
		}

		Lifelong_Machine_Learning_Topic_Model r;
		dataIn d;
		TopicWords t(recordSMatrixWay);
		util u;

		bool recordSMatrix = false;

		string outputFolder = folderPath + folderPathSplit + "LMLTopicModel-" + "alpha=" + u.toString(alpha) + ",beta=" + u.toString(beta) + ",lamda=" + u.toString(lamda) + ",gamma=" + u.toString(gamma) + ",eta=" + u.toString(eta) + ",K=" + u.toString(K) + ",iter=" + to_string(iterNum) + ",topK=" + to_string(topK) + "-splitDatasetWay" + to_string(splitDatasetWay) + "-recordSMatrixWay" + to_string(t.getWay());
#ifdef Linux
		int mkdirStatus = mkdir((outputFolder).c_str(), 0777);
		if (mkdirStatus != 0)
		{
			cout << "Mkdir Error!!! Folder path is " << outputFolder << endl;
			return;
		}
#else
		CreateDirectory((outputFolder).c_str(), NULL);
#endif // Linux

		//d.save(RecordInfo::WMatrix, outputFolder + folderPathSplit + "WMatrix" + ".txt");

		for (int dNum = 1; dNum <= DocNum; dNum++)
		{
			RecordInfo::clearAllWithoutGlobalType();
			RecordInfo::timeRecordStart();
			int subIterNum = 0;
			dMatrix D = d.load(subTfidfFilePath[dNum - 1]);
			D = ~D;
			//r.running(D, UInit[dNum - 1], VInit[dNum - 1], K, dNum, folderPath, 0, 0, 0, 0, iterNum, 0, topK, outputFolder, wordmapFilePath, subWordmapFilePath[dNum - 1], subLabelFilePath[dNum - 1]); //baseline
			dMatrix UAfterNMF = r.running(D, UInit[dNum - 1], VInit[dNum - 1], K, dNum, folderPath, alpha, beta, lamda, gamma, eta, iterNum, subIterNum++, topK, outputFolder, wordmapFilePath, subWordmapFilePath[dNum - 1], subLabelFilePath[dNum - 1], recordSMatrixWay);
			
			int knowledgeGain = 0;
			if (recordSMatrix == true) {
				knowledgeGain = t.recordSMatrix(UAfterNMF, wordmapFilePath, subWordmapFilePath[dNum - 1], topK); //learning knowledge base
				cout << subTfidfFilePath[dNum - 1] + " knowledgeGain is:: " << knowledgeGain << endl;
				//d.save(RecordInfo::SMatrix, outputFolder + folderPathSplit + "SMatrix" + to_string(K) + "_" + to_string(dNum) + "_" + to_string(subIterNum - 1) + ".txt");
			}
			if (knowledgeGain >= knowledgeGainLimit)
			{
				//对以前的语料迭代重新学习
				for (int dNumBP = 1; dNumBP < dNum; dNumBP++)
				{
					for (int iterNumNow = 1; iterNumNow <= iterNumBP; iterNumNow++)
					{
						//以前的语料在新知识下重新NMF
						RecordInfo::clearAllWithoutGlobalType();
						D = d.load(subTfidfFilePath[dNumBP - 1]);
						D = ~D;
						UAfterNMF = r.running(D, UInit[dNumBP - 1], VInit[dNumBP - 1], K, dNum, folderPath, alpha, beta, lamda, gamma, eta, iterNum, subIterNum++, topK, outputFolder, wordmapFilePath, subWordmapFilePath[dNumBP - 1], subLabelFilePath[dNumBP - 1], recordSMatrixWay);
						knowledgeGain = t.recordSMatrix(UAfterNMF, wordmapFilePath, subWordmapFilePath[dNumBP - 1], topK); //learning knowledge base
						cout << subTfidfFilePath[dNumBP - 1] + " knowledgeGain is:: " << knowledgeGain << endl;
						//d.save(RecordInfo::SMatrix, outputFolder + folderPathSplit + "SMatrix" + to_string(K) + "_" + to_string(dNum) + "_" + to_string(subIterNum - 1) + ".txt");
						if (knowledgeGain < knowledgeGainLimit)
							break;

						//当前的语料在新知识下重新NMF
						RecordInfo::clearAllWithoutGlobalType();
						D = d.load(subTfidfFilePath[dNum - 1]);
						D = ~D;
						UAfterNMF = r.running(D, UInit[dNum - 1], VInit[dNum - 1], K, dNum, folderPath, alpha, beta, lamda, gamma, eta, iterNum, subIterNum++, topK, outputFolder, wordmapFilePath, subWordmapFilePath[dNum - 1], subLabelFilePath[dNum - 1], recordSMatrixWay);
						knowledgeGain = t.recordSMatrix(UAfterNMF, wordmapFilePath, subWordmapFilePath[dNum - 1], topK); //learning knowledge base
						cout << subTfidfFilePath[dNum - 1] + " knowledgeGain is:: " << knowledgeGain << endl;
						//d.save(RecordInfo::SMatrix, outputFolder + folderPathSplit + "SMatrix" + to_string(K) + "_" + to_string(dNum) + "_" + to_string(subIterNum - 1) + ".txt");
						if (knowledgeGain < knowledgeGainLimit)
							break;
					}
				}
			}
			RecordInfo::timeRecordEnd();
		}
		if (recordSMatrix == true) {
			d.save2DArrayInLinkStructure(RecordInfo::SMatrix, RecordInfo::SMatrixSize, outputFolder + folderPathSplit + "SMatrix" + to_string(K) + ".txt");
		}
		d.save(RecordInfo::eachDocAllTime, outputFolder + folderPathSplit + "eachDocAllTime.txt");
	}
};