#include "RecordInfo.h"
#include <set>
#include "data.cpp"

RecordInfo::RecordInfo()
{
}

RecordInfo::~RecordInfo()
{
}

double RecordInfo::getSMatrixValue(int rowIndex, int columnIndex) //index start from 0
{
	for (int i = 0; i < SMatrix[rowIndex].size(); i++)
	{
		if (SMatrix[rowIndex][i].index - 1 == columnIndex)
		{
			return SMatrix[rowIndex][i].count;
		}
	}
	//donot exist
	return 0.0;
}

void RecordInfo::setSMatrixValue(int rowIndex, int columnIndex, double value) //index start from 0
{
	for (int i = 0; i < SMatrix[rowIndex].size(); i++)
	{
		if (SMatrix[rowIndex][i].index - 1 == columnIndex)
		{
			SMatrix[rowIndex][i].count = value;
			return;
		}
	}
	//donot exist
	SMatrixNode ss;
	ss.index = columnIndex + 1;
	ss.count = value;
	SMatrix[rowIndex].push_back(ss);
}

dMatrix RecordInfo::generateWMatrix(string labelFilePath)
{
	dataIn d;
	vector<string> label = d.readFile(labelFilePath);
	dMatrix WMatrix(label.size(), label.size());
	for (int i = 0; i < label.size(); i++)
	{
		for (int j = 0; j < label.size(); j++)
		{
			if (label[i].compare(label[j]) == 0)
			{
				WMatrix(i, j) = 1;
			}
			else
			{
				WMatrix(i, j) = 0;
			}
		}
	}
	return WMatrix;
}

void RecordInfo::setMatrixZero(dMatrix &X)
{
	for (int i = 0; i < X.rowno(); i++)
	{
		for (int j = 0; j < X.colno(); j++)
		{
			X(i, j) = 0;
		}
	}
}

void RecordInfo::setMatrixZero(double **X, int rowNum, int ColumnNum)
{
	for (int i = 0; i < rowNum; i++)
	{
		for (int j = 0; j < ColumnNum; j++)
		{
			X[i][j] = 0;
		}
	}
}

void RecordInfo::setMatrixZero(vector<SMatrixNode>* X, int rowNum)
{
	for (int i = 0; i < rowNum; i++)
	{
		X[i].clear();
	}
}

void RecordInfo::getSubAMatrix(const dMatrix &subS, dMatrix &subA)
{
	//generate subA matrix
	for (int i = 0; i < subS.rowno(); i++)
	{
		double sum = 0.0;
		for (int j = 0;j < subS.colno(); j++)
		{
			sum += subS(i, j);
			subA(i, j) = 0;
		}
		subA(i, i) = sum;
	}
}

void RecordInfo::getSubSMatrix(string wordmapFilePath, string subWordmapFilePath, dMatrix &subS)
{
	dataIn d;
	vector<string> wordmap = d.readFile(wordmapFilePath);
	set<string> subWordMapSet = d.readFileForSet(subWordmapFilePath);
	vector<bool> wordmapCheckExist;
	for (int i = 0; i < wordmap.size(); i++)
	{
		if (subWordMapSet.find(wordmap[i]) != subWordMapSet.end())
		{
			wordmapCheckExist.push_back(true);
		}
		else
		{
			wordmapCheckExist.push_back(false);
		}
	}
	//generate subS matrix
	int rowIndex = 0, columnIndex = 0;
	for (int i = 0; i < wordmapCheckExist.size(); i++)
	{
		if (wordmapCheckExist[i])
		{
			columnIndex = 0;
			for (int j = 0; j < wordmapCheckExist.size(); j++)
			{
				if (wordmapCheckExist[j])
				{
					subS(rowIndex, columnIndex) = RecordInfo::getSMatrixValue(i, j);
					columnIndex++;
				}
			}
			rowIndex++;
		}
	}
	//归一化
	double mmax = 0.0;
	for (int i = 0; i < subS.rowno(); i++)
	{
		for (int j = 0; j < subS.colno(); j++)
		{
			if (i != j && subS(i, j) > mmax)
			{
				mmax = subS(i, j);
			}
		}
	}
	mmax += eps;
	for (int i = 0; i < subS.rowno(); i++)
	{
		for (int j = 0; j < subS.colno(); j++)
		{
			if (i == j)
			{
				subS(i, j) = mmax;
			}
			subS(i, j) /= mmax;
		}
	}
}

void RecordInfo::getSubSMatrixWithSmooth(string wordmapFilePath, string subWordmapFilePath, dMatrix &subS)
{
	getSubSMatrix(wordmapFilePath, subWordmapFilePath, subS);
	for (int i = 0; i < subS.rowno(); i++)
	{
		for (int j = 0; j < subS.colno(); j++)
		{
			if (i == j)
			{
				subS(i, j) = 1.0;
			}
			else
			{
				if (subS(i, j) > 0)
				{
					subS(i, j) /= 1.0 / (1 + exp(-2.0*subS(i, j)));
				}
			}
		}
	}
}

void RecordInfo::setTrUTLUInEachUpdateU(const dMatrix &U, const dMatrix &A, const dMatrix &S)
{
	dMatrix L = A - S;
	dMatrix UTLU = mul(mul(~U, L), U);
	double sum = 0.0;
	for (int i = 0; i < UTLU.rowno(); i++)
	{
		sum += UTLU(i, i);
	}
	trUTLUInEachUpdateU.push_back(sum);
}

void RecordInfo::setDiffBetweenDAndUVInEachUpdateU(const dMatrix &U, const dMatrix &D, const dMatrix &V)
{
	int M = D.rowno();
	int N = D.colno();
	dMatrix UV = mul(U, V);
	double sum = 0.0;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			sum += (D(i, j) - UV(i, j))*(D(i, j) - UV(i, j));
		}
	}
	diffBetweenDAndUVInEachUpdateU.push_back(sum);
}

void RecordInfo::setDiffBetweenUTUAndIInEachUpdateUIter(const dMatrix &U)
{
	int K = U.colno();
	dMatrix UT = ~U;
	dMatrix UTU = mul(UT, U);
	double sum = 0.0;
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < K; j++)
		{
			if (i == j)
			{
				UTU(i, j) -= 1.0;
			}
			sum += UTU(i, j)*UTU(i, j);
		}
	}
	diffBetweenUTUAndIInEachUpdateUIter.push_back(sum);
}

void RecordInfo::setDiffBetweenUTDAndVInEachUpdateUIter(const dMatrix &U, const dMatrix &D, const dMatrix &V)
{
	int K = V.rowno();
	int N = V.colno();
	dMatrix UT = ~U;
	dMatrix UTD = mul(UT, D);
	double sum = 0.0;
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < N; j++)
		{
			sum += (UTD(i, j) - V(i, j))*(UTD(i, j) - V(i, j));
		}
	}
	diffBetweenUTDAndVInEachUpdateUIter.push_back(sum);
}

void RecordInfo::setUSparsenessInEachUpdateUIter(const dMatrix &U)
{
	double sparseness = calSparsenessOfMatrix(U);
	uSparsenessInEachUpdateUIter.push_back(sparseness);
}

void RecordInfo::setVSparsenessInEachUpdateVIter(const dMatrix &V)
{
	double sparseness = calSparsenessOfMatrix(V);
	vSparsenessInEachUpdateVIter.push_back(sparseness);
}

void RecordInfo::setOrthogonalityInEachUpdateUIter(const dMatrix &U)
{
	//列归一化
	dMatrix UAfter = U;
	int M = UAfter.rowno();
	int K = UAfter.colno();
	for (int k = 0; k < K; k++)
	{
		double denominator = 0.0;
		for (int m = 0; m < M; m++)
		{
			denominator += UAfter(m, k) * UAfter(m, k);
		}
		denominator = sqrt(denominator);
		if (denominator <= 1e-6)
		{
			denominator = 1e-6;
		}
		for (int m = 0; m < M; m++)
		{
			UAfter(m, k) = UAfter(m, k) / denominator;
		}
	}
	//求正交性
	dMatrix UtAfter = ~UAfter;
	dMatrix UtU = mul(UtAfter, UAfter);
	for (int i = 0; i < K; i++)
	{
		UtU(i, i) = UtU(i, i) - 1.0;
	}
	double ogonality = 0.0;
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < K; j++)
		{
			ogonality += UtU(i, j) * UtU(i, j);
		}
	}
	ogonality = sqrt(ogonality / (K*K));
	orthogonalityInEachUpdateUIter.push_back(ogonality);
}

void RecordInfo::setDiffInEachUpdateUIter(double diff)
{
	diffInEachUpdateUIter.push_back(diff);
}

void RecordInfo::setUpdateUIterNum(int num)
{
	updateUIterNum.push_back(num);
}

void RecordInfo::timeRecordStart(bool allOrNot)
{
	if (allOrNot == true)
	{
		tStartAll = clock();
	}
	else
	{
		tStart = clock();
	}
}

void RecordInfo::timeRecordEnd(bool allOrNot)
{
	if (allOrNot == true)
	{
		tEndAll = clock();
		double runningTime = double(tEndAll - tStartAll) / CLOCKS_PER_SEC; //s
		updateUAllTime.push_back(runningTime);
	}
	else
	{
		tEnd = clock();
		double runningTime = double(tEnd - tStart) / CLOCKS_PER_SEC; //s
		updateUEachIterTime.push_back(runningTime);
	}
}

void RecordInfo::timeRecordStart()
{
	tStartAllEachDoc = clock();
}

void RecordInfo::timeRecordEnd()
{
	tEndAllEachDoc = clock();
	double runningTime = double(tEndAllEachDoc - tStartAllEachDoc) / CLOCKS_PER_SEC; //s
	cout << "running_time: " << runningTime << " s" << endl;
	eachDocAllTime.push_back(runningTime);
}

dMatrix RecordInfo::mul(const dMatrix &A, const dMatrix &B)
{
	dMatrix C(A.rowno(), B.colno());
	#pragma omp parallel for
	for (int i = 0; i < A.rowno(); i++)
	{
		for (int j = 0; j < B.colno(); j++)
		{
			C(i, j) = 0.0;
			for (int k = 0; k < A.colno(); k++)
			{
				C(i, j) += A(i, k) * B(k, j);
			}
		}
	}
	return C;
}

double RecordInfo::calSparsenessOfMatrix(const dMatrix &X)
{
	int allNum = X.rowno() * X.colno();
	int zeroNum = 0;
	for (int i = 0; i < X.rowno(); i++)
	{
		for (int j = 0; j < X.colno(); j++)
		{
			if (X(i, j) <= 0.0)
				zeroNum++;
		}
	}
	return zeroNum * 1.0 / allNum;
}

void RecordInfo::clearAllWithoutGlobalType()
{
	tStart = 0;
	tEnd = 0;
	tStartAll = 0;
	tEndAll = 0;
	updateUIterNum.clear();
	diffInEachUpdateUIter.clear();
	updateUAllTime.clear();
	updateUEachIterTime.clear();
	orthogonalityInEachUpdateUIter.clear();
	uSparsenessInEachUpdateUIter.clear();
	vSparsenessInEachUpdateVIter.clear();
	diffBetweenUTDAndVInEachUpdateUIter.clear();
	diffBetweenUTUAndIInEachUpdateUIter.clear();
	diffBetweenDAndUVInEachUpdateU.clear();
	trUTLUInEachUpdateU.clear();
}

void RecordInfo::clearAllGlobalType()
{
	//clear SMatrix
	for (int i = 0; i < SMatrixSize; i++)
	{
		SMatrix[i].clear();
	}
	eachDocAllTime.clear();
}

time_t RecordInfo::tStart = 0;
time_t RecordInfo::tEnd = 0;
time_t RecordInfo::tStartAll = 0;
time_t RecordInfo::tEndAll = 0;
time_t RecordInfo::tStartAllEachDoc = 0;
time_t RecordInfo::tEndAllEachDoc = 0;
double RecordInfo::eps = 1e-8;
vector<int> RecordInfo::updateUIterNum; //updateU过程中的迭代次数
vector<double> RecordInfo::diffInEachUpdateUIter; //每个updateU内部迭代的diff
vector<double> RecordInfo::eachDocAllTime; //each batch过程的时间
vector<double> RecordInfo::updateUAllTime; //updateU过程的时间
vector<double> RecordInfo::updateUEachIterTime; //updateU过程的内部迭代的时间
vector<double> RecordInfo::orthogonalityInEachUpdateUIter; //updateU过程的内部迭代的U矩阵的正交性
vector<double> RecordInfo::uSparsenessInEachUpdateUIter; //updateU过程的内部迭代的U矩阵的稀疏度
vector<double> RecordInfo::vSparsenessInEachUpdateVIter; //updateV过程的内部迭代的V矩阵的稀疏度
vector<double> RecordInfo::diffBetweenUTDAndVInEachUpdateUIter; //updateU过程的内部迭代中 U^T*D-V的F范数的平方 的值
vector<double> RecordInfo::diffBetweenUTUAndIInEachUpdateUIter; //updateU过程的内部迭代中 U^T*U-I的F范数的平方 的值
vector<double> RecordInfo::diffBetweenDAndUVInEachUpdateU; //updateU过程中 D-U*V的F范数的平方 的值
vector<double> RecordInfo::trUTLUInEachUpdateU; //updateU过程中tr(U^T*L*U) 的值

//global type, no need to reset
vector<SMatrixNode>* RecordInfo::SMatrix; //M*Len, 链式存储, 防止M太大导致内存不足, term A与term B是一个主题的topK, 则 SMatrix(A, B) += 1, 之后再进行归一化
int RecordInfo::SMatrixSize = 0;
//dMatrix RecordInfo::AMatrix; //M*M, diag(S*向量(1)), 即对角元素为S矩阵相应的行的元素之和

//dMatrix RecordInfo::WMatrix; //N*N, 文档A与文档B是同一个label则 WMatrix(A, B) = 1, 否则为0
//dMatrix RecordInfo::TMatrix; //N*N, diag(W*向量(1)), 即对角元素为W矩阵相应的行的元素之和
