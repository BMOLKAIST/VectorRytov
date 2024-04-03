
#include "cuComplex.h"
#include <complex.h>


/*
* Add a constant to a vector.
*/
__device__ int modulo(int m, int n) { return m >= 0 ? m % n : (n - abs(m % n)) % n; }

__device__ __forceinline__ cuComplex cexpf(cuComplex z)
{

	cuComplex res;
	float t = expf(z.x);
	sincosf(z.y, &res.y, &res.x);
	res.x *= t;
	res.y *= t;
	return res;

}

__device__ float2  mat3_MULT_coeff(float2* MAT1, float2* MAT2, int i, int j) {
	float2 val = make_cuFloatComplex(0, 0);
	for (int k = 0; k < 3; k++) {
		val = cuCaddf(val, cuCmulf(MAT1[k * 3 + i], MAT2[j * 3 + k]));
	}
	return val;
}
__device__ float2  mat3_MULT_coeff(float2* MAT1, float* MAT2, int i, int j) {
	float2 val = make_cuFloatComplex(0, 0);
	for (int k = 0; k < 3; k++) {
		val = cuCaddf(val, cuCmulf(MAT1[k * 3 + i], make_cuFloatComplex(MAT2[j * 3 + k],0)));
	}
	return val;
}
__device__ float2  mat3_MULT_coeff(float* MAT1, float2* MAT2, int i, int j) {
	float2 val = make_cuFloatComplex(0, 0);
	for (int k = 0; k < 3; k++) {
		val = cuCaddf(val, cuCmulf(make_cuFloatComplex(MAT1[k * 3 + i],0), MAT2[j * 3 + k]));
	}
	return val;
}

struct convolve_params {
	int pos_A_shared;
	int pos_B_shared;
	int pos_Uz_shared;
	int pos_V0_shared;
	int pos_Vout_shared;
	int size_2D;
	int size_z;
	int size_angle;
};
__global__ void convolve_kernel(float const* const A_global, float const* const B_global, float const* const Uz_global, float2 const* const V0_global, float2* Vout_global, 
	int pos_A_shared,
	int pos_B_shared,
	int pos_Uz_shared,
	int pos_V0_shared,
	int pos_Vout_shared,
	int pos_Vp_shared,
	int pos_temp_shared,
	int pos_tmat_shared,
	int size_2D,
	int size_z,
	int size_angle,
	float res_z, 
	float start_z)
{
	extern __shared__ float data_shared[];

	int column_num = blockIdx.x; // the global column of data
	int vertical_index = threadIdx.x; //the vertical position of the data

	float* A = &data_shared[pos_A_shared];
	float* B = &data_shared[pos_B_shared];
	float* Uz = &data_shared[pos_Uz_shared];
	float2* V0 = (float2*)(&data_shared[pos_V0_shared]);
	float2* Vout = (float2*)(&data_shared[pos_Vout_shared]); 
	float2* Vp = (float2*)(&data_shared[pos_Vp_shared]);
	float2* temp = (float2*)(&data_shared[pos_temp_shared]);
	float2* tmat= (float2*)(&data_shared[pos_tmat_shared]);

	//load data 
	if (column_num < size_2D) {//verify we are working on a valid column
		
		int start_field = 0;
		while (start_field + vertical_index < size_angle * 9) {

			A[start_field + vertical_index] = A_global[9 * column_num + (start_field + vertical_index) % 9 + ((start_field + vertical_index) / 9) * size_2D * 9];
			B[start_field + vertical_index] = B_global[(start_field + vertical_index) % 9 + ((start_field + vertical_index) / 9) * 9];
			V0[start_field + vertical_index] = V0_global[9 * column_num + (start_field + vertical_index) % 9 + ((start_field + vertical_index) / 9) * size_2D * 9];

			start_field = start_field + size_z;
		}
		
		start_field = 0;
		while (start_field + vertical_index < size_angle) {
			Uz[start_field + vertical_index] = Uz_global[column_num + (start_field + vertical_index) * size_2D];

			start_field = start_field + size_z;
		}

		if (vertical_index < size_z) {
			for (int k = 0; k < 9; k++) {
				Vout[vertical_index + k * size_z] = make_cuFloatComplex(0, 0);
			}
		}
		start_field = 0;
		while (start_field + vertical_index < size_z * 9) {
			Vp[start_field + vertical_index]=Vout_global[9 * column_num + (start_field + vertical_index) % 9 + ((start_field + vertical_index) / 9) * size_2D * 9];
			start_field = start_field + size_z;
		}
		

		//summ
		for (int angle = 0; angle < size_angle; angle++) {

			__syncthreads();

			int vertical_skip = ((size_z) / 9);
			int mat_id = vertical_index % 9;
			int vert_id = vertical_index / 9;

			if (vertical_index < vertical_skip * 9) {

				temp[vertical_index] = make_cuFloatComplex(0, 0);



				//sum like some in parallel
				for (int k = 0; vert_id + k * vertical_skip < size_z; k++) {
					int curr_index = mat_id + (vert_id + k * vertical_skip) * 9;
					int z = (vert_id + k * vertical_skip);
					if (curr_index < size_z * 9) {
						float2 uz_val = make_cuFloatComplex(0,- 6.283185307179586f * (Uz[angle]) * (float(z) * res_z + start_z));
						temp[vertical_index] = cuCaddf(temp[vertical_index],
							cuCmulf(Vp[curr_index], cexpf(uz_val))
						);
					}
				}
			}
			__syncthreads();

			//sum the rest in series
			if (vert_id == 0) {
				for (int k = 1; k < vertical_skip; k++) {
					temp[vertical_index] = cuCaddf(temp[vertical_index], temp[vertical_index + k * 9]);
				}
				temp[vertical_index] = cuCsubf(temp[vertical_index], V0[angle * 9 + vertical_index]);
			}
			__syncthreads();
			if (vert_id == 0) {
				tmat[vertical_index] = mat3_MULT_coeff(&A[angle * 9], &temp[0], vertical_index % 3, vertical_index / 3);
			}
			__syncthreads();
			if (vert_id == 0) {
				temp[vertical_index] = mat3_MULT_coeff(&tmat[0], &B[angle * 9], vertical_index % 3, vertical_index / 3);
			}

			// resumation 

			__syncthreads();
			start_field = 0;
			while (start_field + vertical_index < size_z *9) {
				int z = (start_field + vertical_index)/9;
				float2 uz_val = make_cuFloatComplex(0, 6.283185307179586f * (Uz[angle]) * (float(z) * res_z + start_z));
				Vout[start_field + vertical_index] = cuCaddf(Vout[start_field + vertical_index], cuCmulf(temp[(start_field + vertical_index) % 9], cexpf(uz_val)));
				start_field = start_field + size_z;
			}
			
		}
		//save data 
		__syncthreads();
		start_field = 0;
		while (start_field + vertical_index < size_z * 9) {
			int mat_pos = (start_field + vertical_index) % 9;
			int mat_pos_trans = (mat_pos % 3) * 3 + (mat_pos / 3);
			float2 val_symetric = cuCaddf(Vout[start_field + vertical_index], Vout[start_field + vertical_index - mat_pos + mat_pos_trans]);
			//val_symetric = Vout[start_field + vertical_index];
			Vout_global[9 * column_num + (start_field + vertical_index) % 9 + ((start_field + vertical_index) / 9) * size_2D * 9] = cuCmulf(val_symetric, make_cuFloatComplex(0.5,0));
			
			start_field = start_field + size_z;
		}
	}
}
