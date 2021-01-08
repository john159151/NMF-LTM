#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <set>

#include "matrix.h"
#include "util.cpp"
#include "SMatrixNode.cpp"


using namespace std;

typedef techsoft::matrix<double> dMatrix;

#pragma once
class dataIn
{
public:
	dataIn() {}
	~dataIn() {}
	
	void generateMatrix(dMatrix &X, double value, string filePath) //定值生成
	{
		for (int i = 0; i < X.rowno(); i++)
		{
			for (int j = 0; j < X.colno(); j++)
			{
				X(i, j) = value;
			}
		}
		save(X, filePath);
	}

	void generateMatrix(dMatrix &X, double from, double to, string filePath) //随机生成
	{
		X.rand(from, to);
		save(X, filePath);
	}

	/**
	* load data from given file (default format) to matrix
	*/
	dMatrix load(string filepath)
	{
		ifstream fin;
		fin.open(filepath.c_str(), ios::in);
		string line;

		int lineNo = 0;
		int termNo = 0;

		util u;//

			   // get the rows and columns corresponding to lineNO and termNo
		while (!fin.eof())
		{
			getline(fin, line);
			line = filterEnter(line);

			// cout<<"read from file "<<line<<endl;

			if (line.compare("") == 0)
			{
				cout << "empty" << endl;
			}
			else
			{
				lineNo++;
				vector<string> vec = u.split(line, " ");

				for (int i = 0; i<vec.size(); i++)
				{
					string temp = vec[i];
					if (!temp.compare("") == 0)
					{
						vector<string> termValuePair = u.split(vec[i], ":");
						int termIndex = atoi((termValuePair[0]).c_str()); // extract the index of term: from 1,2,...
						if (termIndex>termNo)
						{
							termNo = termIndex;
						}
					}
				}
			}


		}

		fin.close();

		// reback to original stats
		fin.open(filepath.c_str(), ios::in); //
		dMatrix dm(lineNo, termNo);

		cout << "line NO: " << lineNo << endl;
		cout << "term NO: " << termNo << endl;

		int lineNO2 = 0;

		while (!fin.eof())
		{
			getline(fin, line);
			line = filterEnter(line);

			if (line.compare("") == 0)
			{
				cout << "done" << endl;
			}
			else
			{
				lineNO2++;

				vector<string> listPair = u.split(line, " ");
				for (int i = 0; i<listPair.size(); i++)
				{
					string tempstr = listPair[i];
					if (!tempstr.compare("") == 0)
					{
						vector<string> values = u.split(listPair[i], ":");

						int index = atoi((values[0]).c_str());
						double val = atof((values[1]).c_str());

						//cout<<"index: "<<index<<"---"<<"val: "<<val<<endl;

						dm(lineNO2 - 1, index - 1) = val;
					}
				}
			}

		}

		return dm;
	}

	/**
	* load data from given file (default format) to matrix with given dimension
	*/
	dMatrix load(string filepath, int M, int N)
	{
		cout << "filepath is:: " << filepath << endl;
		ifstream fin;
		fin.open(filepath.c_str(), ios::in);
		dMatrix dm(M, N);
		string line;
		util u;

		int lineNO2 = 0;

		while (!fin.eof())
		{
			getline(fin, line);
			line = filterEnter(line);

			if (line.compare("") == 0)
			{
				cout << "done" << endl;
			}
			else
			{
				lineNO2++;

				vector<string> listPair = u.split(line, " ");
				for (int i = 0; i<listPair.size(); i++)
				{
					string tempstr = listPair[i];
					if (!tempstr.compare("") == 0)
					{
						vector<string> values = u.split(listPair[i], ":");

						int index = atoi((values[0]).c_str());
						double val = atof((values[1]).c_str());

						//cout<<"index: "<<index<<"---"<<"val: "<<val<<endl;

						dm(lineNO2 - 1, index - 1) = val;
					}
				}
			}

		}

		return dm;
	}

	/**
	* load data from given file (default format) to two-dimmensional array with given dimension
	*/
	double** load2DArray(string filepath, int M, int N)
	{
		cout << "filepath is:: " << filepath << endl;
		ifstream fin;
		fin.open(filepath.c_str(), ios::in);
		double **dm = new double*[M];
		for (int i = 0; i < M; i++)
		{
			dm[i] = new double[N];
		}
		string line;
		util u;

		int lineNO2 = 0;

		while (!fin.eof())
		{
			getline(fin, line);
			line = filterEnter(line);

			if (line.compare("") == 0)
			{
				cout << "done" << endl;
			}
			else
			{
				lineNO2++;

				vector<string> listPair = u.split(line, " ");
				for (int i = 0; i<listPair.size(); i++)
				{
					string tempstr = listPair[i];
					if (!tempstr.compare("") == 0)
					{
						vector<string> values = u.split(listPair[i], ":");

						int index = atoi((values[0]).c_str());
						double val = atof((values[1]).c_str());

						//cout<<"index: "<<index<<"---"<<"val: "<<val<<endl;

						dm[lineNO2 - 1][index - 1] = val;
					}
				}
			}

		}

		return dm;
	}

	/**
	* load data from given file (default format) to link structure with given dimension
	*/
	vector<SMatrixNode>* load2DArrayInLinkStructure(string filepath, int M)
	{
		cout << "filepath is:: " << filepath << endl;
		ifstream fin;
		fin.open(filepath.c_str(), ios::in);
		vector<SMatrixNode>* dm = new vector<SMatrixNode>[M];
		for (int i = 0; i < M; i++)
		{
			dm[i].clear();
		}
		string line;
		util u;

		int lineNO2 = 0;

		while (!fin.eof())
		{
			getline(fin, line);
			line = filterEnter(line);

			if (line.compare("") == 0)
			{
				cout << "done" << endl;
			}
			else
			{
				lineNO2++;

				vector<string> listPair = u.split(line, " ");
				for (int i = 0; i<listPair.size(); i++)
				{
					string tempstr = listPair[i];
					if (!tempstr.compare("") == 0)
					{
						vector<string> values = u.split(listPair[i], ":");

						int index = atoi((values[0]).c_str());
						double val = atof((values[1]).c_str());
						
						SMatrixNode ss;
						ss.index = index;
						ss.count = val;
						dm[lineNO2 - 1].push_back(ss);
					}
				}
			}

		}

		return dm;
	}

	/**
	* read file info from given file
	* return string vector
	* 注意：读取UTF-8编码文件会乱码，需要将所有文件都统一编码！！！这样即使乱码了，字符串比较的时候也不会出错
	*/
	vector<string> readFile(string filepath)
	{
		ifstream fin;
		fin.open(filepath.c_str(), ios::in);
		string line;
		int lineNum = 0;
		vector<string> info;
		while (!fin.eof())
		{
			getline(fin, line);
			line = filterEnter(line);
			if (fin.eof())
			{
				break;
			}
			lineNum++;
			if (line.compare("") == 0)
			{
				cout << "line " << lineNum << " empty but had calculate" << endl;
				info.push_back(line);
			}
			else
			{
				info.push_back(line);
			}
		}
		fin.close();
		return info;
	}

	/**
	* read file info from given file
	* return string set
	*/
	set<string> readFileForSet(string filepath)
	{
		vector<string> info = readFile(filepath);
		set<string> infoSet;
		for (int i = 0; i < info.size(); i++)
		{
			infoSet.insert(info[i]);
		}
		return infoSet;
	}

	/**
	* get file line number from given file
	*/
	int getLineNumFromFile(string filepath)
	{
		vector<string> info = readFile(filepath);
		return info.size();
	}


	/**
	* write file info to the given file
	*/
	void writeFile(vector<string> info, string filepath)
	{
		ofstream fout;
		fout.open(filepath.c_str(), ios::out);

		if (!fout)   // if the file could open, hFile is a handle, else is zero
		{
			cout << "fail..." << endl;
			return;
		}

		for (int i = 0; i<info.size(); i++)
		{
			fout << info[i] << "\n";
		}

		fout.close();
	}


	/**
	* save matrix to file with the default format: sparse format
	*/
	void save(dMatrix m, string filepath)
	{
		ofstream fout;
		fout.open(filepath.c_str(), ios::out);

		if (!fout)   // if the file could open, hFile is a handle, else is zero
		{
			cout << "fail..." << endl;
			return;
		}

		int rows = m.rowno();
		int cols = m.colno();

		for (int i = 0; i<rows; i++)
		{
			string lineStr; //

			for (int j = 0; j<cols; j++)
			{

				if (m(i, j) != 0)
				{
					char s1[60];
					char s2[60];

					int index = j + 1;
					sprintf(s1, "%d", index);
					string sIndex(s1);

					double vtemp = m(i, j);
					sprintf(s2, "%.9f", vtemp);
					string sValue(s2);

					lineStr += (sIndex + ":" + sValue + " ");
				}
			}

			//cout<<lineStr<<endl;
			fout << lineStr << "\n";

		}

		fout.close();

		cout << "The matrix has been saved successfully." << endl;
	}

	/**
	* save two-dimensional array to file with the default format: sparse format
	*/
	void save2DArray(double** m, int rowNum, int columnNum, string filepath)
	{
		ofstream fout;
		fout.open(filepath.c_str(), ios::out);

		if (!fout)   // if the file could open, hFile is a handle, else is zero
		{
			cout << "fail..." << endl;
			return;
		}

		int rows = rowNum;
		int cols = columnNum;

		for (int i = 0; i<rows; i++)
		{
			string lineStr; //

			for (int j = 0; j<cols; j++)
			{

				if (m[i][j] != 0)
				{
					char s1[60];
					char s2[60];

					int index = j + 1;
					sprintf(s1, "%d", index);
					string sIndex(s1);

					double vtemp = m[i][j];
					sprintf(s2, "%.9f", vtemp);
					string sValue(s2);

					lineStr += (sIndex + ":" + sValue + " ");
				}
			}

			//cout<<lineStr<<endl;
			fout << lineStr << "\n";

		}

		fout.close();

		cout << "The matrix has been saved successfully." << endl;
	}

	/**
	* save link structure array to file with the default format: sparse format
	*/
	void save2DArrayInLinkStructure(vector<SMatrixNode>* m, int rowNum, string filepath)
	{
		ofstream fout;
		fout.open(filepath.c_str(), ios::out);

		if (!fout)   // if the file could open, hFile is a handle, else is zero
		{
			cout << "fail..." << endl;
			return;
		}

		int rows = rowNum;

		for (int i = 0; i<rows; i++)
		{
			string lineStr = "";

			for (int j = 0; j < m[i].size(); j++)
			{
				char s1[60];
				char s2[60];

				int index = m[i][j].index;
				sprintf(s1, "%d", index);
				string sIndex(s1);

				double vtemp = m[i][j].count;
				sprintf(s2, "%.9f", vtemp);
				string sValue(s2);

				lineStr += (sIndex + ":" + sValue + " ");
			}

			//cout<<lineStr<<endl;
			fout << lineStr << "\n";

		}

		fout.close();

		cout << "The matrix has been saved successfully." << endl;
	}

	/**
	* save vector to file with the default format
	*/
	template <typename T>
	void save(vector<T> info, string filepath)
	{
		ofstream fout;
		fout.open(filepath.c_str(), ios::out);

		if (!fout)   // if the file could open, hFile is a handle, else is zero
		{
			cout << "fail..." << endl;
			return;
		}

		for (int i = 0; i < info.size(); i++)
		{
			string lineStr = getValueInString(info[i]);
			fout << lineStr << "\n";
		}

		fout.close();

		cout << "The vector has been saved successfully." << endl;
	}

private:
	string getValueInString(int x)
	{
		char s[60];
		sprintf(s, "%d", x);
		string sStr(s);
		return sStr;
	}

	string getValueInString(double x)
	{
		char s[60];
		sprintf(s, "%.9f", x);
		string sStr(s);
		return sStr;
	}

	string filterEnter(string line) //把string最后的\r\n去掉
	{
		for (int i = line.size() - 1; i >= 0; i--)
		{
			if (line[i] != '\n' && line[i] != '\r')
			{
				return line.substr(0, i + 1);
			}
		}
		return line;
	}

};
