#define _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <math.h>

#define N 1000
#define tau 0.01
#define max_pos 5.0
#define min_pos -5.0

char fname[] = "results[n = 1000].txt";

const double tmax = 1.0;
double t = 0.0;

using namespace std;

__global__ void Kernel(double* U, double* U_new) {

	int A1 = 1, A2 = 2;
	int p1 = 3, p2 = 2;

	double sum_Vx = 0.0, sum_Vy = 0.0;

	__shared__ double temp[4 * N];

	int i = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

	if (i < N * 4)
	{
		temp[i + 2] = U[i + 2];
		temp[i + 3] = U[i + 3];
	}

	__syncthreads();

	if (i < N * 4)
	{
		for (int j = 0; j < N * 4; j += 4)
		{
			if (i != j)
			{

				temp[i] = (A1 * (temp[j + 2] - temp[i + 2])) / pow(sqrt(pow(temp[j + 2] - temp[i + 2], 2) + pow(temp[j + 3] - temp[i + 3], 2)), p1) -
					(A2 * (temp[j + 2] - temp[i + 2])) / pow(sqrt(pow(temp[j + 2] - temp[i + 2], 2) + pow(temp[j + 3] - temp[i + 3], 2)), p2);

				temp[i + 1] = (A1 * (temp[j + 3] - temp[i + 3])) / pow(sqrt(pow(temp[j + 2] - temp[i + 2], 2) + pow(temp[j + 3] - temp[i + 3], 2)), p1) -
					(A2 * (temp[j + 3] - temp[i + 3])) / pow(sqrt(pow(temp[j + 2] - temp[i + 2], 2) + pow(temp[j + 3] - temp[i + 3], 2)), p2);

				sum_Vx += temp[i];
				sum_Vy += temp[i + 1];

				U_new[i] = temp[i] + tau * sum_Vx;
				U_new[i + 1] = temp[i + 1] + tau * sum_Vy;

				double tmp_x = temp[i + 2] + tau * U_new[i];
				double tmp_y = temp[i + 3] + tau * U_new[i + 1];

				if ((tmp_x > max_pos) || (tmp_x < min_pos))
				{
					U_new[i] = -U_new[i];
					U_new[i + 2] = temp[i + 2] + tau * U_new[i];
				}
				else
				{
					U_new[i + 2] = tmp_x;
				}

				if ((tmp_y > max_pos) || (tmp_y < min_pos))
				{
					U_new[i + 1] = -U_new[i + 1];
					U_new[i + 3] = temp[i + 3] + tau * U_new[i + 1];
				}
				else
				{
					U_new[i + 3] = tmp_y;
				}
			}
		}

		temp[i] = U_new[i];
		temp[i + 1] = U_new[i + 1];
		temp[i + 2] = U_new[i + 2];
		temp[i + 3] = U_new[i + 3];
	}
}

int main() {

	ofstream file(fname);

	if (file.is_open()) {

		cudaEvent_t tn, tk;
		float runtime = 0.0;

		int size = sizeof(double) * 4 * N;

		double* U = new double[4 * N];

		for (int i = 0; i < (4 * N); i += 4)
		{
			U[i + 2] = (double)(rand()) / RAND_MAX - 0.5;
			U[i + 3] = (double)(rand()) / RAND_MAX - 0.5;
		}

		double* Unew_Dev = NULL;
		double* U_Dev = NULL;

		cudaMalloc((void**)&Unew_Dev, size);
		cudaMalloc((void**)&U_Dev, size);

		cudaEventCreate(&tn);
		cudaEventCreate(&tk);

		cudaEventRecord(tn, 0);

		cudaMemcpy(U_Dev, U, size, cudaMemcpyHostToDevice);

		do {

			Kernel << < 1, N >> > (U_Dev, Unew_Dev);

			cudaThreadSynchronize();

			cudaMemcpy(U, Unew_Dev, size, cudaMemcpyDeviceToHost);

			file << "t = " << t << endl;
			for (int i = 0; i < (4 * N); i += 4)
			{
				file << "\tU[" << U[i + 2] << ", " << U[i + 3] << "]" << endl;
			}

			t += tau;

		} while (t < tmax);

		cudaEventRecord(tk, 0);
		cudaEventSynchronize(tk);
		cudaEventElapsedTime(&runtime, tn, tk);

		file << endl << "runtime = " << runtime / 1000.0 << " sec" << endl;

		delete[] U;
		cudaFree(Unew_Dev);
		cudaFree(U_Dev);
	}
	else {
		cout << "can't open file " << fname << endl;
	}

	return 0;
}