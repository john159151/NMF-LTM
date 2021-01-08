#include "stdio.h"
#include <iostream>
#include <string>
#include "data.cpp"
#include "matrix.cpp"
#include "util.cpp"
#include <omp.h>
#include "RecordInfo.h"
#include "SplitD.cpp"
#include "Test.cpp"
#include "LMLTM_Framework.cpp"
 
//#define TESTING

using namespace std;

int main()
{
#ifdef TESTING
	Test tt;
	tt.running();
	while (true);
#endif

#ifdef Linux
	const string folderPathSplit = "/";
#else
	const string folderPathSplit = "\\";
#endif

#ifdef Linux
	string folderPath = "/home/user/John/LMLTM_Amazon1k_K50_DocNum50_filter2/"; //Amazon1k
#else
	string folderPath = "E:\\学习\\科研\\Lifelone Machine Learning\\running";
#endif

	string sourceFileName = "Amazon1k"; //Amazon1k

	const bool needToSplitData = false;
	const bool needToGenerateUVInitMatrix = false;
	const bool needToUseNewSMatrix = true;

	double alpha, beta, lamda, gamma, eta;
	int topK = 10; //每个topic选前topK个词语更新S矩阵
	int iterNum = 100; //每个NMF updateU, updateV的最大迭代次数
	int iterNumBP = 0; //反馈式对以前语料重新学习的最大迭代次数(③->④->①->②)
	int knowledgeGainLimit = 0; //S矩阵有<knowledgeGainLimit个元素从0变非0说明S矩阵已达稳定
	/*
	way1: 每个sub-dataset包含所有label的data(如第一个dataset包含第1、2、3、4、5类data，第二个dataset包含第1、2、3、4、5类data); 
	way2: 每个sub-dataset包含不同label的data(如第一个dataset包含第1、2类data，第二个dataset包含第3、4、5类data); 
	way3: 每个sub-dataset包含不同label且只包含一个label的data(如第一、二、三个dataset包含第1类data，第四、五、六个dataset包含第2类data); 
	way4: 构造5个特定sub-dataset，前3个为第一类的，后两个是第二类、第三类; 
	way5: 构造7个特定sub-dataset，前5个为第一类的，后两个是第二类、第三类; 
	way6: 构造7个特定sub-dataset，前5个为第一类的，后两个是第二类、第三类; 源dataset每类数据数量不相同;
	way7: 构造DocNum个特定sub-dataset，所有都为第一类，每类eachLabelNum个; 源dataset每类数据数量不相同;
	way8: 构造DocNum个特定sub-dataset，每个sub-dataset包含所有属于同一个label的数据;
	way9: 构造DocNum个特定sub-dataset，按顺序划分，类别混杂，每个sub-dataset最多eachLabelNum个;
	way10: 构造DocNum个特定sub-dataset，按派牌方式划分（伪随机），类别混杂，每个sub-dataset最多eachLabelNum个;
	way11: 构造DocNum个特定sub-dataset，按顺序划分，类别混杂，每个sub-dataset最多eachLabelNum个，最后一个sub-dataset取剩余的所有;
	*/
	const int splitDatasetWay = 7; 
	const int filterSubDatasetFrequency = 1; //split小的dataset后，过滤掉每个sub-dataset中df小于filterSubDatasetFrequency的term
	const int recordSMatrixWay = 1; //way1: 通过每对terms pair加1，然后归一化生成SMatrix; way2: 通过每对terms pair置为1(不管出现多少次)，然后生成SMatrix; way3: way1累加再平滑

	dataIn d;
	string wordmapFilePath = folderPath + folderPathSplit + "wordmap.txt";
	string labelFilePath = folderPath + folderPathSplit + "label.txt";
	int M = d.getLineNumFromFile(wordmapFilePath);
	int N = d.getLineNumFromFile(folderPath + folderPathSplit + sourceFileName + ".txt");
	RecordInfo::SMatrixSize = M;
	vector<string> labelInfo = d.readFile(labelFilePath);

	int DocNum = 50; //必须要对文档平均分, 因为下一语料的U, V需要用到上一次语料的U, V, 用way11的方法划分也是可以的
	int eachLabelNum = 1000; //仅用于划分数据集

	vector<string> subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath;
	for (int i = 1; i <= DocNum; i++)
	{
		subWordmapFilePath.push_back(folderPath + folderPathSplit + "wordmap_" + sourceFileName + "_" + to_string(i) + ".txt");
		subTfidfFilePath.push_back(folderPath + folderPathSplit + "TFIDF_" + sourceFileName + "_" + to_string(i) + ".txt");
		subSourceFilePath.push_back(folderPath + folderPathSplit + sourceFileName + "_" + to_string(i) + ".txt");
		subLabelFilePath.push_back(folderPath + folderPathSplit + "label" + "_" + to_string(i) + ".txt");
	}
	SplitD s(splitDatasetWay, filterSubDatasetFrequency);
	if (needToSplitData == true) 
	{
		s.splitDataset(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
	}

	if (needToUseNewSMatrix == true) 
	{
		RecordInfo::SMatrix = new vector<SMatrixNode>[RecordInfo::SMatrixSize];
		RecordInfo::setMatrixZero(RecordInfo::SMatrix, RecordInfo::SMatrixSize);
		//RecordInfo::WMatrix = RecordInfo::generateWMatrix(labelFilePath);
	}
	else
	{
		RecordInfo::SMatrix = d.load2DArrayInLinkStructure(folderPath + folderPathSplit + "SMatrix_Old2.txt", M);
	}
	cout << "SMatrix init done" << endl;

	int K[] = { 50 };
	//#pragma omp parallel for //不支持多线程并发(因为涉及RecordInfo的static变量)
	for (int j = 0; j < sizeof(K)/sizeof(K[0]); j++)
	{
		//对于每个subDataset创建固定的随机初始U, V矩阵
		vector<string> UInitFilePath, VInitFilePath;
		for (int i = 1; i <= DocNum; i++)
		{
			UInitFilePath.push_back(folderPath + folderPathSplit + "UInit" + "_K" + to_string(K[j]) + "_" + to_string(i) + ".txt");
			VInitFilePath.push_back(folderPath + folderPathSplit + "VInit" + "_K" + to_string(K[j]) + "_" + to_string(i) + ".txt");
		}
		if (needToGenerateUVInitMatrix == true) 
		{
			for (int dNum = 1; dNum <= DocNum; dNum++)
			{
				int M = d.getLineNumFromFile(subWordmapFilePath[dNum - 1]);
				int N = d.getLineNumFromFile(subTfidfFilePath[dNum - 1]);
				dMatrix UTemp(M, K[j]), VTemp(K[j], N);
				//U, V初始化为0.5的时候效果不好, P, R, F1降到0.4+
				//U, V初始化为1.0的时候效果不好, P, R, F1降到0.5+
				d.generateMatrix(UTemp, RecordInfo::eps, 1, UInitFilePath[dNum - 1]);
				d.generateMatrix(VTemp, RecordInfo::eps, 1, VInitFilePath[dNum - 1]);
			}
		}

		//读入U, V初始化矩阵
		vector<dMatrix>UInit, VInit;
		for (int dNum = 1; dNum <= DocNum; dNum++)
		{
			UInit.push_back(d.load(UInitFilePath[dNum - 1]));
			VInit.push_back(d.load(VInitFilePath[dNum - 1]));
		}

		LMLTM_Framework l;
		for (int recordSMatrixWayNow = recordSMatrixWay; recordSMatrixWayNow >= recordSMatrixWay; recordSMatrixWayNow--)
		{
			alpha = 10; beta = 0.5; lamda = 0.001; gamma = 0.001; eta = 1;
			l.running(alpha, beta, lamda, gamma, eta, folderPath, folderPathSplit,
				iterNum, iterNumBP, topK, knowledgeGainLimit, DocNum, wordmapFilePath,
				UInit, VInit, recordSMatrixWayNow, K[j], subTfidfFilePath, subWordmapFilePath, subLabelFilePath, splitDatasetWay, needToUseNewSMatrix);
		}
	}

	//delete RecordInfo::SMatrix space
	delete[] RecordInfo::SMatrix;
	
}