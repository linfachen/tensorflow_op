#define blocksize 256
#define gridsize 32


__global__ void mul(const float* A,const float * B, float * C, int n)
{
	int loop = n/(blocksize*gridsize);
	for(int i =0;i<loop;i++){
		*(C + i*(blocksize*gridsize) + (256 *blockIdx.x + threadIdx.x)) =  \
		*(C + i*(blocksize*gridsize) + (256 *blockIdx.x + threadIdx.x)) *  \
		*(C + i*(blocksize*gridsize) + (256 *blockIdx.x + threadIdx.x));
	} 
	if((256 *blockIdx.x + threadIdx.x)<n%(blocksize*gridsize)){
		*(C + loop*(blocksize*gridsize) + (256 *blockIdx.x + threadIdx.x)) =  \
		*(C + loop*(blocksize*gridsize) + (256 *blockIdx.x + threadIdx.x)) *  \
		*(C + loop*(blocksize*gridsize) + (256 *blockIdx.x + threadIdx.x));		
	}	
}


void eltwise_mul(const float* A,const float * B, float * C, int n)
{
	mul<<<gridsize, blocksize>>>(A, B, C, n);
}
