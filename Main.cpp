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
	string folderPath = "E:\\ѧϰ\\����\\Lifelone Machine Learning\\running";
#endif

	string sourceFileName = "Amazon1k"; //Amazon1k

	const bool needToSplitData = false;
	const bool needToGenerateUVInitMatrix = false;
	const bool needToUseNewSMatrix = true;

	double alpha, beta, lamda, gamma, eta;
	int topK = 10; //ÿ��topicѡǰtopK���������S����
	int iterNum = 100; //ÿ��NMF updateU, updateV������������
	int iterNumBP = 0; //����ʽ����ǰ��������ѧϰ������������(��->��->��->��)
	int knowledgeGainLimit = 0; //S������<knowledgeGainLimit��Ԫ�ش�0���0˵��S�����Ѵ��ȶ�
	/*
	way1: ÿ��sub-dataset��������label��data(���һ��dataset������1��2��3��4��5��data���ڶ���dataset������1��2��3��4��5��data); 
	way2: ÿ��sub-dataset������ͬlabel��data(���һ��dataset������1��2��data���ڶ���dataset������3��4��5��data); 
	way3: ÿ��sub-dataset������ͬlabel��ֻ����һ��label��data(���һ����������dataset������1��data�����ġ��塢����dataset������2��data); 
	way4: ����5���ض�sub-dataset��ǰ3��Ϊ��һ��ģ��������ǵڶ��ࡢ������; 
	way5: ����7���ض�sub-dataset��ǰ5��Ϊ��һ��ģ��������ǵڶ��ࡢ������; 
	way6: ����7���ض�sub-dataset��ǰ5��Ϊ��һ��ģ��������ǵڶ��ࡢ������; Դdatasetÿ��������������ͬ;
	way7: ����DocNum���ض�sub-dataset�����ж�Ϊ��һ�࣬ÿ��eachLabelNum��; Դdatasetÿ��������������ͬ;
	way8: ����DocNum���ض�sub-dataset��ÿ��sub-dataset������������ͬһ��label������;
	way9: ����DocNum���ض�sub-dataset����˳�򻮷֣������ӣ�ÿ��sub-dataset���eachLabelNum��;
	way10: ����DocNum���ض�sub-dataset�������Ʒ�ʽ���֣�α������������ӣ�ÿ��sub-dataset���eachLabelNum��;
	way11: ����DocNum���ض�sub-dataset����˳�򻮷֣������ӣ�ÿ��sub-dataset���eachLabelNum�������һ��sub-datasetȡʣ�������;
	*/
	const int splitDatasetWay = 7; 
	const int filterSubDatasetFrequency = 1; //splitС��dataset�󣬹��˵�ÿ��sub-dataset��dfС��filterSubDatasetFrequency��term
	const int recordSMatrixWay = 1; //way1: ͨ��ÿ��terms pair��1��Ȼ���һ������SMatrix; way2: ͨ��ÿ��terms pair��Ϊ1(���ܳ��ֶ��ٴ�)��Ȼ������SMatrix; way3: way1�ۼ���ƽ��

	dataIn d;
	string wordmapFilePath = folderPath + folderPathSplit + "wordmap.txt";
	string labelFilePath = folderPath + folderPathSplit + "label.txt";
	int M = d.getLineNumFromFile(wordmapFilePath);
	int N = d.getLineNumFromFile(folderPath + folderPathSplit + sourceFileName + ".txt");
	RecordInfo::SMatrixSize = M;
	vector<string> labelInfo = d.readFile(labelFilePath);

	int DocNum = 50; //����Ҫ���ĵ�ƽ����, ��Ϊ��һ���ϵ�U, V��Ҫ�õ���һ�����ϵ�U, V, ��way11�ķ�������Ҳ�ǿ��Ե�
	int eachLabelNum = 1000; //�����ڻ������ݼ�

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
	//#pragma omp parallel for //��֧�ֶ��̲߳���(��Ϊ�漰RecordInfo��static����)
	for (int j = 0; j < sizeof(K)/sizeof(K[0]); j++)
	{
		//����ÿ��subDataset�����̶��������ʼU, V����
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
				//U, V��ʼ��Ϊ0.5��ʱ��Ч������, P, R, F1����0.4+
				//U, V��ʼ��Ϊ1.0��ʱ��Ч������, P, R, F1����0.5+
				d.generateMatrix(UTemp, RecordInfo::eps, 1, UInitFilePath[dNum - 1]);
				d.generateMatrix(VTemp, RecordInfo::eps, 1, VInitFilePath[dNum - 1]);
			}
		}

		//����U, V��ʼ������
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