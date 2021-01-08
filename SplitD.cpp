#include <iostream>
#include <math.h>
#include "matrix.h"
#include <omp.h>
#include <stdio.h>
#include <exception>
#include "RecordInfo.h"
#include "data.cpp"
#include <vector>
#include <map>
#include <set>
#include <math.h>

using namespace std;
typedef techsoft::matrix<double> dMatrix;
typedef std::valarray<double> dVector;

#pragma once
class SplitD
{
public:
	//constructors
	SplitD(int wayChoose, int filterNumChoose)
	{
		way = wayChoose;
		filterNum = filterNumChoose;
	}

	~SplitD() {}

	void splitDataset(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo)
	{
		if (way == 1)
		{
			splitDatasetInsideEachLabel(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 2)
		{
			splitDatasetWithDifferentLabel(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 3)
		{
			splitDatasetWithDifferentLabelAndOnlyOneLabelInDataset(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 4)
		{
			splitDatasetWith5Dataset(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 5)
		{
			splitDatasetWith7Dataset(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 6)
		{
			splitDatasetWith7DatasetInDifferenceNumberOfLabel(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 7)
		{
			splitDatasetWithSameLabelDatasetInDifferenceNumberOfLabel(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 8)
		{
			splitDatasetWithOneLabelInOneDatasetInDifferenceNumberOfLabel(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 9)
		{
			splitDatasetInInputOrderInDifferenceNumberOfLabel(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 10)
		{
			splitDatasetInSendCardOrderInDifferenceNumberOfLabel(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else if (way == 11)
		{
			splitDatasetInInputOrderInDifferenceNumberOfLabelWithAll(folderPath, sourceFileName, DocNum, eachLabelNum, wordmapFilePath, subWordmapFilePath, subTfidfFilePath, subSourceFilePath, subLabelFilePath, labelInfo);
		}
		else
		{
			cout << "Split dataset way " + to_string(way) + " is not define!";
		}
	}

	int getWay()
	{
		return way;
	}

private:
	void splitDatasetInsideEachLabel(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //ÿһ��label����ƽ���ֿ�DocNum���֣����ÿ����dataset����ÿ��label��һС����
	{
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		int labelNum = info.size() / eachLabelNum;
		vector<string> wordmap = d.readFile(wordmapFilePath);

		for (int docNum = 1; docNum <= DocNum; docNum++)
		{
			//generate sub-Dataset source and sub-label
			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int label = 1; label <= labelNum; label++)
			{
				for (int index = (label - 1)*eachLabelNum + (docNum - 1)*(eachLabelNum / DocNum); index < (label - 1)*eachLabelNum + (docNum)*(eachLabelNum / DocNum); index++)
				{
					dataset.push_back(info[index]);
					subLabel.push_back(labelInfo[index]);
				}
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum-1]);
		}
	}

	void splitDatasetWithDifferentLabel(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //ÿһ��label����ƽ���ֿ�DocNum���֣����ÿ����dataset����ÿ��label��һС����
	{
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		int labelNum = info.size() / eachLabelNum;
		vector<string> wordmap = d.readFile(wordmapFilePath);

		int labelNumLeft = labelNum;
		for (int docNum = 1; docNum <= DocNum; docNum++)
		{
			//generate sub-Dataset source and sub-label
			int labelFrom = labelNum - labelNumLeft + 1, labelTo = labelFrom + labelNumLeft / (DocNum - docNum + 1) - 1;
			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = (labelFrom - 1)*eachLabelNum; index < (labelTo)*eachLabelNum; index++)
			{
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);
			labelNumLeft -= labelNumLeft / (DocNum - docNum + 1);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);
		}
	}

	void splitDatasetWithDifferentLabelAndOnlyOneLabelInDataset(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //ÿ��sub-dataset������ͬlabel��ֻ����һ��label��data(���һ����������dataset������1��data�����ġ��塢����dataset������2��data)
	{
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		int labelNum = info.size() / eachLabelNum;
		vector<string> wordmap = d.readFile(wordmapFilePath);

		int datasetNumInEachLabel = DocNum / labelNum; //ÿ��label�зֳ�dataset������
		int lineNumInEachSubDataset = 150; //ÿ��sub-dataset���ĵ���
		for (int docNum = 1; docNum <= DocNum; docNum++)
		{
			//generate sub-Dataset source and sub-label
			int labelNow = docNum / datasetNumInEachLabel + (docNum%datasetNumInEachLabel == 0 ? 0 : 1);
			int labelFrom = (labelNow - 1)*eachLabelNum + ((docNum - 1) % datasetNumInEachLabel)*lineNumInEachSubDataset;
			int labelTo = labelFrom + lineNumInEachSubDataset;

			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = labelFrom; index < labelTo; index++)
			{
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);
		}
	}

	void splitDatasetWith5Dataset(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //����5���ض�sub-dataset��ǰ3��Ϊ��һ��ģ��������ǵڶ��ࡢ������
	{
		DocNum = 5;
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		int labelNum = info.size() / eachLabelNum;
		vector<string> wordmap = d.readFile(wordmapFilePath);

		int lineNumInEachSubDataset = 150; //ÿ��sub-dataset���ĵ���
		for (int docNum = 1; docNum <= DocNum; docNum++)
		{
			//generate sub-Dataset source and sub-label
			int labelFrom = (docNum - 1) * lineNumInEachSubDataset;
			if (docNum >= 4)
			{
				labelFrom = (docNum - 3)*eachLabelNum;
			}
			int labelTo = labelFrom + lineNumInEachSubDataset;

			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = labelFrom; index < labelTo; index++)
			{
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);
		}
	}

	void splitDatasetWith7Dataset(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //����7���ض�sub-dataset��ǰ5��Ϊ��һ��ģ��������ǵڶ��ࡢ������
	{
		DocNum = 7;
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		int labelNum = info.size() / eachLabelNum;
		vector<string> wordmap = d.readFile(wordmapFilePath);

		int lineNumInEachSubDataset = 100; //ÿ��sub-dataset���ĵ���
		for (int docNum = 1; docNum <= DocNum; docNum++)
		{
			//generate sub-Dataset source and sub-label
			int labelFrom = (docNum - 1) * lineNumInEachSubDataset;
			if (docNum >= 6)
			{
				labelFrom = (docNum - 5)*eachLabelNum;
			}
			int labelTo = labelFrom + lineNumInEachSubDataset;

			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = labelFrom; index < labelTo; index++)
			{
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);
		}
	}

	void splitDatasetWith7DatasetInDifferenceNumberOfLabel(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //����7���ض�sub-dataset��ǰ5��Ϊ��һ��ģ��������ǵڶ��ࡢ������
	{
		DocNum = 7;
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		vector<string> wordmap = d.readFile(wordmapFilePath);

		int lineNumInEachSubDataset = eachLabelNum; //ÿ��sub-dataset���ĵ���
		for (int docNum = 1; docNum <= DocNum; docNum++)
		{
			//generate sub-Dataset source and sub-label
			int labelFrom = (docNum - 1) * lineNumInEachSubDataset;
			if (docNum >= 6)
			{
				for (int i = 6; i <= docNum; i++)
				{
					labelFrom++;
					while (labelInfo[labelFrom].compare(labelInfo[labelFrom - 1]) == 0)
					{
						labelFrom++;
					}
				}
			}
			int labelTo = labelFrom + lineNumInEachSubDataset;

			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = labelFrom; index < labelTo; index++)
			{
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);
		}
	}

	void splitDatasetWithSameLabelDatasetInDifferenceNumberOfLabel(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //����DocNum���ض�sub-dataset�����ж�Ϊ��һ�࣬ÿ��eachLabelNum��; Դdatasetÿ��������������ͬ;
	{
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		vector<string> wordmap = d.readFile(wordmapFilePath);

		int lineNumInEachSubDataset = eachLabelNum; //ÿ��sub-dataset���ĵ���
		for (int docNum = 1; docNum <= DocNum; docNum++)
		{
			//generate sub-Dataset source and sub-label
			int labelFrom = (docNum - 1) * lineNumInEachSubDataset;
			int labelTo = labelFrom + lineNumInEachSubDataset;

			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = labelFrom; index < labelTo; index++)
			{
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);
		}
	}

	void splitDatasetWithOneLabelInOneDatasetInDifferenceNumberOfLabel(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //����DocNum���ض�sub-dataset�����ж�Ϊ��һ�࣬ÿ��eachLabelNum��; Դdatasetÿ��������������ͬ;
	{
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		vector<string> wordmap = d.readFile(wordmapFilePath);
		int labelIndexNow = 0;

		//int lineNumInEachSubDataset = eachLabelNum; //ÿ��sub-dataset���ĵ���
		for (int docNum = 1; docNum <= DocNum; docNum++)
		{
			//generate sub-Dataset source and sub-label
			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = labelIndexNow; index < labelInfo.size(); index++)
			{
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
				if (index + 1 < labelInfo.size() && labelInfo[index + 1] != labelInfo[labelIndexNow])
				{
					labelIndexNow = index + 1;
					break;
				}
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);
		}
	}
		
	void splitDatasetInInputOrderInDifferenceNumberOfLabel(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //����DocNum���ض�sub-dataset�����ж�Ϊ��һ�࣬ÿ��eachLabelNum��; Դdatasetÿ��������������ͬ;
	{
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		vector<string> wordmap = d.readFile(wordmapFilePath);
		int labelIndexNow = 0;

		int lineNumInEachSubDataset = eachLabelNum; //ÿ��sub-dataset���ĵ���
		int lineNow = 0;
		int docNum = 1;
		while (lineNow < info.size())
		{
			//generate sub-Dataset source and sub-label
			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = lineNow; index < lineNow + lineNumInEachSubDataset; index++)
			{
				if (index >= info.size())
				{
					break;
				}
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);

			lineNow += lineNumInEachSubDataset;
			docNum++;
		}
	}

	void splitDatasetInInputOrderInDifferenceNumberOfLabelWithAll(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo) //����DocNum���ض�sub-dataset�����ж�Ϊ��һ�࣬ÿ��eachLabelNum��; Դdatasetÿ��������������ͬ�����һ��sub-dataset��������ʣ�µ�;
	{
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		vector<string> wordmap = d.readFile(wordmapFilePath);
		int labelIndexNow = 0;

		int lineNumInEachSubDataset = eachLabelNum; //ÿ��sub-dataset���ĵ���
		int lineNow = 0;
		int docNum = 1;
		while (lineNow < info.size())
		{
			//generate sub-Dataset source and sub-label
			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = lineNow; index < lineNow + lineNumInEachSubDataset; index++)
			{
				if (index >= info.size())
				{
					break;
				}
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
			}
			if (docNum == DocNum) //���һ��sub-dataset����ʣ�µ�ȫ��
			{
				for (int index = lineNow + lineNumInEachSubDataset; index < info.size(); index++)
				{
					dataset.push_back(info[index]);
					subLabel.push_back(labelInfo[index]);
				}
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);

			lineNow += lineNumInEachSubDataset;
			if (docNum == DocNum)
				lineNow = info.size();
			docNum++;
		}
	}

	void splitDatasetInSendCardOrderInDifferenceNumberOfLabel(string folderPath, string sourceFileName, int DocNum, int eachLabelNum, string wordmapFilePath, const vector<string> &subWordmapFilePath, const vector<string> &subTfidfFilePath, const vector<string> &subSourceFilePath, const vector<string> &subLabelFilePath, const vector<string> &labelInfo)
	{
		string sourceFilePath = folderPath + folderPathSplit + sourceFileName + ".txt";
		dataIn d;
		vector<string> info = d.readFile(sourceFilePath); //all data
		vector<string> wordmap = d.readFile(wordmapFilePath);
		int labelIndexNow = 0;

		int lineNumInEachSubDataset = eachLabelNum; //ÿ��sub-dataset���ĵ���
		for (int docNum = 1; docNum <= DocNum; docNum++)
		{
			//generate sub-Dataset source and sub-label
			vector<string> dataset; //sub-dataset
			vector<string> subLabel; //sub-label
			for (int index = docNum-1; index < info.size(); index+=DocNum)
			{
				//cout << docNum << " " << index << endl;
				if (dataset.size() >= lineNumInEachSubDataset)
				{
					break;
				}
				dataset.push_back(info[index]);
				subLabel.push_back(labelInfo[index]);
			}
			filterDocFrequency(dataset, subLabel);
			d.writeFile(dataset, subSourceFilePath[docNum - 1]);
			d.writeFile(subLabel, subLabelFilePath[docNum - 1]);

			//generate info
			generateInfoForEachDataset(dataset, wordmap, subWordmapFilePath[docNum - 1], subTfidfFilePath[docNum - 1]);

		}
	}

	void generateInfoForEachDataset(vector<string> &dataset, vector<string> &wordmapAll, string subWordmapFilePath, string tfidfFilePath)
	{
		dataIn d;
		util u;
		//generate wordmap in each sub-Dataset
		set<string> wordsetInSubDataset;
		for (int i = 0; i < dataset.size(); i++)
		{
			vector<string> words = u.split(dataset[i], " ");
			for (int j = 0; j < words.size(); j++)
			{
				wordsetInSubDataset.insert(words[j]);
			}
		}
		vector<string> wordmapInSubDataset;
		for (int i = 0; i < wordmapAll.size(); i++)
		{
			if (wordsetInSubDataset.find(wordmapAll[i]) != wordsetInSubDataset.end())
			{
				wordmapInSubDataset.push_back(wordmapAll[i]);
			}
		}
		d.writeFile(wordmapInSubDataset, subWordmapFilePath);

		//generate TFIDF in each sub-Dataset
		int subM = wordmapInSubDataset.size();
		dMatrix tfValue = solveTf(dataset, wordmapInSubDataset);
		dMatrix idfValue = solveIdf(dataset, wordmapInSubDataset);
		dMatrix tfidfValue(dataset.size(), subM); //subN * subM
		for (int i = 0; i < tfValue.rowno(); i++)
		{
			for (int j = 0; j < tfValue.colno(); j++)
			{
				tfidfValue(i, j) = tfValue(i, j)*idfValue(i, j);
			}
		}
		d.save(tfidfValue, tfidfFilePath);
	}

	dMatrix solveTf(vector<string> info, vector<string> wordmap)
	{
		util u;
		map<string, int> wordIndex;
		for (int i = 0; i < wordmap.size(); i++)
		{
			wordIndex[wordmap[i]] = i;
		}
		dMatrix tfValue(info.size(), wordmap.size());
		for (int n = 0; n < info.size(); n++)
		{
			vector<string> words = u.split(info[n], " ");
			map<string, int> wordCount;
			int sum = 0;
			for (int i = 0; i<words.size(); i++)
			{
				if (!words[i].compare("") == 0)
				{
					if (wordCount.find(words[i]) != wordCount.end())
					{
						wordCount[words[i]] = wordCount[words[i]] + 1;
					}
					else
					{
						wordCount[words[i]] = 1;
					}
					sum++;
				}
			}
			for (int i = 0; i<words.size(); i++)
			{
				if (!words[i].compare("") == 0)
				{
					double tf = wordCount[words[i]] * 1.0 / sum;
					tfValue(n, wordIndex[words[i]]) = tf;
				}
			}
		}
		return tfValue;
	}

	dMatrix solveIdf(vector<string> info, vector<string> wordmap)
	{
		util u;
		map<string, int> wordIndex;
		map<string, int> wordCount;
		for (int i = 0; i < wordmap.size(); i++)
		{
			wordIndex[wordmap[i]] = i;
			wordCount[wordmap[i]] = 0;
		}
		dMatrix idfValue(info.size(), wordmap.size());
		for (int n = 0; n < info.size(); n++)
		{
			set<string> wordsSetInEachDoc;
			vector<string> words = u.split(info[n], " ");
			for (int i = 0; i < words.size(); i++)
			{
				if (!words[i].compare("") == 0)
				{
					if (wordsSetInEachDoc.find(words[i]) == wordsSetInEachDoc.end())
					{
						wordsSetInEachDoc.insert(words[i]);
						wordCount[words[i]]++;
					}
				}
			}
		}
		for (int n = 0; n < info.size(); n++)
		{
			vector<string> words = u.split(info[n], " ");
			for (int i = 0; i < words.size(); i++)
			{
				if (!words[i].compare("") == 0)
				{
					double idf = log(info.size()*1.0 / (wordCount[words[i]]+1));
					idfValue(n, wordIndex[words[i]]) = idf;
				}
			}
		}
		return idfValue;
	}

	void filterDocFrequency(vector<string> &dataset, vector<string> &subLabel) //��>=filterNum���ĵ��г��ֵ�terms�Żᱣ��
	{
		util u;
		map<string, int> wordCount;
		for (int n = 0; n < dataset.size(); n++)
		{
			set<string> wordsSetInEachDoc;
			vector<string> words = u.split(dataset[n], " ");
			for (int i = 0; i < words.size(); i++)
			{
				if (!words[i].compare("") == 0)
				{
					if (wordsSetInEachDoc.find(words[i]) == wordsSetInEachDoc.end())
					{
						wordsSetInEachDoc.insert(words[i]);
						wordCount[words[i]]++;
					}
				}
			}
		}
		for (int n = 0; n < dataset.size(); n++)
		{
			vector<string> words = u.split(dataset[n], " ");
			string lineInfo = "";
			for (int i = 0; i < words.size(); i++)
			{
				if (!words[i].compare("") == 0 && wordCount[words[i]] >= filterNum)
				{
					if (lineInfo.length() > 0)
					{
						lineInfo += " ";
					}
					lineInfo += words[i];
				}
			}
			dataset[n] = lineInfo;
		}
		for (vector<string>::iterator itDataset = dataset.begin(), itLabel = subLabel.begin(); itDataset != dataset.end() && itLabel != subLabel.end(); )
		{
			if ((*itDataset).length() == 0)
			{
				cout << "erase!" << endl;
				itDataset = dataset.erase(itDataset);
				itLabel = subLabel.erase(itLabel);
				continue;
			}
			itDataset++;
			itLabel++;
		}
	}

	int way;
	int filterNum;
#ifdef Linux
	const string folderPathSplit = "/";
#else
	const string folderPathSplit = "\\";
#endif
};
