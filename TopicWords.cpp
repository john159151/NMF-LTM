#include <iostream>
#include <math.h>
#include "matrix.h"
#include <omp.h>
#include <stdio.h>
#include <exception>
#include "RecordInfo.h"
#include <vector>
#include "data.cpp"
#include <algorithm>
#include <math.h>
#include <map>

using namespace std;

typedef techsoft::matrix<double> dMatrix;
typedef std::valarray<double> dVector;

struct node
{
	int index; //start from 0
	double value;
};

#pragma once
class TopicWords
{
public:
	//constructors
	TopicWords(int wayInput)
	{
		way = wayInput;
	}

	~TopicWords() {}

	int recordSMatrix(const dMatrix &U, string wordMapFilePath, string subWordMapFilePath, int topK)
	{
		if (way == 1)
		{
			return recordSMatrixWithTermsPairAddOne(U, wordMapFilePath, subWordMapFilePath, topK);
		}
		else if (way == 2)
		{
			return recordSMatrixWithTermsPairSetOne(U, wordMapFilePath, subWordMapFilePath, topK);
		}
		else if (way == 3)
		{
			return recordSMatrixWithTermsPairAddOne(U, wordMapFilePath, subWordMapFilePath, topK); //使用时Smooth（1/(1+exp(-2*x))）一下，记录时与Way1一样
		}
		else
		{
			cout << "Record SMatrix way " + to_string(way) + " is not define!";
		}
		return 0;
	}

	vector<vector<string> > getTopicWords(dMatrix &U, int topK, vector<string> wordmap) // K*topK
	{
		vector<vector<string> > result;
		vector<vector<int> > topicWordsIndex = getTopicWordsIndex(U, topK);
		for (int i = 0; i < topicWordsIndex.size(); i++)
		{
			vector<string> eachTopicWords;
			for (int j = 0; j < topicWordsIndex[i].size(); j++)
			{
				eachTopicWords.push_back(wordmap[topicWordsIndex[i][j]]);
			}
			result.push_back(eachTopicWords);
		}
		return result;
	}

	vector<vector<int> > getTopicWordsIndex(const dMatrix &U, int topK) // K*topK
	{
		vector<vector<int> > result;
		vector<node> eachTopicValue;
		for (int k = 0; k < U.colno(); k++)
		{
			eachTopicValue.clear();
			for (int m = 0; m < U.rowno(); m++)
			{
				node p;
				p.index = m;
				p.value = U(m, k);
				eachTopicValue.push_back(p);
			}
			//sort
			sort(eachTopicValue.begin(), eachTopicValue.end(), cmp);
			vector<int> topicWords;
			for (int i = 0; i < topK; i++)
			{
				topicWords.push_back(eachTopicValue[i].index);
			}
			result.push_back(topicWords);
		}
		return result;
	}

	void outputTopic(const vector<vector<int> > &topicWordsIndex, const vector<string> &wordmap)
	{
		cout << "Topic::" << endl;
		for (int i = 0; i < topicWordsIndex.size(); i++)
		{
			for (int j = 0; j < topicWordsIndex[i].size(); j++)
			{
				cout << topicWordsIndex[i][j] << "_" << wordmap[topicWordsIndex[i][j]] << " ";
			}
			cout << endl;
		}
	}

	bool static cmp(node aa, node bb)
	{
		return aa.value > bb.value;
	}

	int getWay()
	{
		return way;
	}

private:
	int way;

	int recordSMatrixWithTermsPairAddOne(const dMatrix &U, string wordMapFilePath, string subWordMapFilePath, int topK) //通过每对terms pair加1，然后归一化生成SMatrix
	{
		int knowledgeGain = 0;
		dataIn d;
		vector<string> wordmap = d.readFile(wordMapFilePath);
		vector<string> subWordmap = d.readFile(subWordMapFilePath);
		map<string, int> wordIndexMap;
		for (int i = 0; i < wordmap.size(); i++)
		{
			wordIndexMap[wordmap[i]] = i;
		}

		vector<vector<int> > topicWordsIndex = getTopicWordsIndex(U, topK);
		//outputTopic(topicWordsIndex, wordmap);
		for (int i = 0; i < topicWordsIndex.size(); i++)
		{
			for (int j = 0; j < topicWordsIndex[i].size(); j++)
			{
				for (int k = j + 1;k < topicWordsIndex[i].size(); k++)
				{
					if (RecordInfo::getSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][j]]], wordIndexMap[subWordmap[topicWordsIndex[i][k]]]) == 0.0)
						knowledgeGain++;
					RecordInfo::setSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][j]]], wordIndexMap[subWordmap[topicWordsIndex[i][k]]], RecordInfo::getSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][j]]], wordIndexMap[subWordmap[topicWordsIndex[i][k]]])+1.0);
					if (RecordInfo::getSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][k]]], wordIndexMap[subWordmap[topicWordsIndex[i][j]]]) == 0.0)
						knowledgeGain++;
					RecordInfo::setSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][k]]], wordIndexMap[subWordmap[topicWordsIndex[i][j]]], RecordInfo::getSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][k]]], wordIndexMap[subWordmap[topicWordsIndex[i][j]]])+1.0);
				}
			}
		}
		//设置对角线元素为全局最大值
		double mmax = 0.0;
		for (int i = 0; i < RecordInfo::SMatrixSize; i++)
		{
			for (int j = 0; j < RecordInfo::SMatrixSize; j++)
			{
				if (i != j && RecordInfo::getSMatrixValue(i, j) > mmax)
				{
					mmax = RecordInfo::getSMatrixValue(i, j);
				}
			}
		}
		mmax += RecordInfo::eps;
		for (int i = 0; i < RecordInfo::SMatrixSize; i++)
		{
			RecordInfo::setSMatrixValue(i, i, mmax);
		}
		return knowledgeGain;
	}

	int recordSMatrixWithTermsPairSetOne(const dMatrix &U, string wordMapFilePath, string subWordMapFilePath, int topK) //通过每对terms pair置为1(不管出现多少次)，然后生成SMatrix
	{
		int knowledgeGain = 0;
		dataIn d;
		vector<string> wordmap = d.readFile(wordMapFilePath);
		vector<string> subWordmap = d.readFile(subWordMapFilePath);
		map<string, int> wordIndexMap;
		for (int i = 0; i < wordmap.size(); i++)
		{
			wordIndexMap[wordmap[i]] = i;
		}

		vector<vector<int> > topicWordsIndex = getTopicWordsIndex(U, topK);
		//outputTopic(topicWordsIndex, wordmap);
		for (int i = 0; i < topicWordsIndex.size(); i++)
		{
			for (int j = 0; j < topicWordsIndex[i].size(); j++)
			{
				for (int k = j + 1;k < topicWordsIndex[i].size(); k++)
				{
					if (RecordInfo::getSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][j]]], wordIndexMap[subWordmap[topicWordsIndex[i][k]]]) == 0.0)
						knowledgeGain++;
					RecordInfo::setSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][j]]], wordIndexMap[subWordmap[topicWordsIndex[i][k]]], 1.0);
					if (RecordInfo::getSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][k]]], wordIndexMap[subWordmap[topicWordsIndex[i][j]]]) == 0.0)
						knowledgeGain++;
					RecordInfo::setSMatrixValue(wordIndexMap[subWordmap[topicWordsIndex[i][k]]], wordIndexMap[subWordmap[topicWordsIndex[i][j]]], 1.0);
				}
			}
		}
		for (int i = 0; i < RecordInfo::SMatrixSize; i++)
		{
			RecordInfo::setSMatrixValue(i, i, 1.0 + RecordInfo::eps);
		}

		return knowledgeGain;
	}

};
