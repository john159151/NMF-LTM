#include <iostream>
#include <math.h>
#include "matrix.h"
#include <omp.h>
#include <stdio.h>
#include <exception>
#include "RecordInfo.h"

using namespace std;

typedef techsoft::matrix<double> dMatrix;
typedef std::valarray<double> dVector;

#pragma once
class LMLTM_Update
{
public:
	//constructors
	LMLTM_Update()
	{
		precison = 1.0e-6;
		maxIter = 100;
	}

	LMLTM_Update(int iterNum)
	{
		precison = 1.0e-6;
		maxIter = iterNum;
	}

	~LMLTM_Update() {}

	//updating functions
	void updateV(const dMatrix &D, const dMatrix &U, dMatrix &V)
	{
		// dimension D:M*N, U:M*K
		int N = D.colno();
		int K = U.colno();

		cout << "N" << N << ":" << "K" << K << endl;

		dMatrix UTD = RecordInfo::mul(~U, D);
		dMatrix UTUV = RecordInfo::mul(RecordInfo::mul(~U, U), V);

		//#pragma omp parallel for
		for (int k = 0; k<K; k++)
		{
			for (int n = 0; n<N; n++)
			{
				V(k, n) = V(k, n)*UTD(k, n) / (UTUV(k, n) + RecordInfo::eps);
			}
		}
		//RecordInfo::setVSparsenessInEachUpdateVIter(V);
	}

	//updating functions
	void updateU(const dMatrix &D, dMatrix &U, const dMatrix &V, double lamda, const dMatrix &subS, const dMatrix &subA)
	{
		// dimension D:M*N, U:M*K
		int M = D.rowno();
		int K = V.rowno();

		cout << "M" << M << ":" << "K" << K << endl;

		dMatrix U_pre = U;
		dMatrix DVT = RecordInfo::mul(D, ~V);
		dMatrix UVVT = RecordInfo::mul(U, RecordInfo::mul(V, ~V));

		dMatrix up = DVT;
		dMatrix down = UVVT;

		if (lamda != 0) //baseline, without knowledge base
		{
			dMatrix SU = RecordInfo::mul(subS, U);
			dMatrix AU = RecordInfo::mul(subA, U);
			up = up + lamda*SU;
			down = down + lamda*AU;
		}

		//#pragma omp parallel for
		for (int m = 0; m<M; m++)
		{
			for (int k = 0; k<K; k++)
			{
				U(m, k) = U(m, k)*up(m, k) / (down(m, k) + RecordInfo::eps);
			}
		}

		double diff = 0.0;
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < K; j++)
			{
				if (fabs(U(i, j) - U_pre(i, j)) > diff)
				{
					diff = fabs(U(i, j) - U_pre(i, j));
				}
			}
		}
		//RecordInfo::setDiffInEachUpdateUIter(diff);
		//RecordInfo::setOrthogonalityInEachUpdateUIter(U);
		//RecordInfo::setUSparsenessInEachUpdateUIter(U);
		//RecordInfo::setDiffBetweenDAndUVInEachUpdateU(U, D, V);
		//RecordInfo::setTrUTLUInEachUpdateU(U, subA, subS);
	}

	// general updating functions
	// D: M*N; U: M*K; V: K*N
	// alpha: denotes the prior knowledge from the past experiences
	// beta: marks the diversity of learned topics; the larger, the more diverse.
	void updateU(const dMatrix &D, dMatrix &U, const dMatrix &V, double alpha, double beta, double eta, const dMatrix &subS, const dMatrix &subA)
	{
		// dimension D:M*N, U:M*K
		int M = D.rowno();
		int K = V.rowno();

		cout << "M" << M << ":" << "K" << K << endl;

		dMatrix U_pre = U;
		dMatrix up = generateZeroMatrix(D.rowno(), V.rowno()); //M*K
		if (eta != 0.0) {
			up = RecordInfo::mul(D, ~V); // up=DVT
		}
		dMatrix down = generateZeroMatrix(U.rowno(), V.rowno()); //M*K
		if (eta != 0.0) {
			down = RecordInfo::mul(U, RecordInfo::mul(V, ~V)); // down=UVVT
		}

		// if alpha is not zero, LTM learns with priors
		if (alpha != 0)
		{
			dMatrix SU = RecordInfo::mul(subS, U);
			dMatrix AU = RecordInfo::mul(subA, U);
			up = up + alpha*SU;
			down = down + alpha*AU;
		}

		// if beta is not zero, LTM learns with diverse topics. (U is orthogonal)
		if (beta != 0)
		{
			up = up + 2 * beta*U;
			down = down + 2 * beta*RecordInfo::mul(U, RecordInfo::mul(~U, U));
		}

		//#pragma omp parallel for
		for (int m = 0; m<M; m++)
		{
			for (int k = 0; k<K; k++)
			{
				if (up(m, k) == 0.0)
				{
					U(m, k) = 0.0;
					continue;
				}
				U(m, k) = U(m, k)*up(m, k) / (down(m, k) + RecordInfo::eps);
			}
		}

		double diff = 0.0;
		for (int i = 0; i<M; i++)
		{
			for (int j = 0; j < K; j++)
			{
				if (fabs(U(i, j) - U_pre(i, j))>diff)
				{
					diff = fabs(U(i, j) - U_pre(i, j));
				}
			}
		}
		//RecordInfo::setDiffInEachUpdateUIter(diff);
		//RecordInfo::setOrthogonalityInEachUpdateUIter(U);
		//RecordInfo::setUSparsenessInEachUpdateUIter(U);
		//RecordInfo::setDiffBetweenDAndUVInEachUpdateU(U, D, V);
		//RecordInfo::setTrUTLUInEachUpdateU(U, subA, subS);
	}


	// general updating functions
	// D: M*N; U: M*K; V: K*N
	// lamda: denotes the sparsity constraints on matrix V; the larger, the more sparse.
	// gamma: marks the label-supervised information.
	void updateV(const dMatrix &D, const dMatrix &U, dMatrix &V, double lamda, double gamma, double eta, const dMatrix &subW, const dMatrix &subT)
	{
		// dimension D:M*N, U:M*K
		int N = D.colno();
		int K = U.colno();

		cout << "N" << N << ":" << "K" << K << endl;

		dMatrix up = generateZeroMatrix(U.colno(), D.colno()); //K*N
		if (eta != 0.0) {
			up = RecordInfo::mul(~U, D); // up=UTD
		}
		dMatrix down = generateZeroMatrix(U.colno(), V.colno()); //K*N
		if (eta != 0.0) {
			down = RecordInfo::mul(RecordInfo::mul(~U, U), V); // down=UTUV
		}

		// if lamda is not zero, LTM learns with sparsity on matrix V.
		if (lamda != 0)
		{
			dMatrix C(V.rowno(), V.colno());
			for (size_t i = 0; i < V.rowno(); i++)
			{
				for (size_t j = 0; j < V.colno(); j++)
				{
					C(i, j) = 1.0;
				}
			}

			down = down + 0.5*lamda*C;
		}

		// if gamma is not zero, LTM learns with class label-supervised information.
		if (gamma != 0)
		{
			up = up + gamma*RecordInfo::mul(V, subW);
			down = down + gamma*RecordInfo::mul(V, subT);

		}

		//#pragma omp parallel for
		for (int k = 0; k<K; k++)
		{
			for (int n = 0; n<N; n++)
			{
				if (up(k, n) == 0.0)
				{
					V(k, n) = 0.0;
					continue;
				}
				V(k, n) = V(k, n)*up(k, n) / (down(k, n) + RecordInfo::eps);
			}
		}
		//RecordInfo::setVSparsenessInEachUpdateVIter(V);
	}

	dMatrix generateZeroMatrix(int rowno, int colno)
	{
		dMatrix C(rowno, colno);
		for (int i = 0; i < rowno; i++)
		{
			for (int j = 0; j < colno; j++)
			{
				C(i, j) = 0.0;
			}
		}
		return C;
	}

private:
	int maxIter;
	double precison;
};
