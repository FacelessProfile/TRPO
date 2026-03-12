//TO COMPILE: gcc -O3 -march=native -mavx2 -mfma -fopenmp syrk.c -lopenblas -o syrksdzc
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <string.h>
#include <math.h>

#define MR 8
#define NR 8
#define MC 256
#define KC 256
#define NC 512

static float* aligned_alloc_f(size_t n){
    void* p;
    if(posix_memalign(&p,64,n*sizeof(float))!=0) exit(1);
    return (float*)p;
}

static inline void pack_A(int mc,int kc,const float *A,int lda,float *Ap){
    for(int k=0;k<kc;k++)
        for(int i=0;i<mc;i++)
            Ap[k*MR+i]=A[i+k*lda];
}

static inline void pack_B(int kc,int nc,const float *B,int ldb,float *Bp){
    for(int k=0;k<kc;k++)
        memcpy(&Bp[k*NR],&B[k*ldb],nc*sizeof(float));
}

static inline void kernel_8x8(int kc,const float* A,const float* B,float* C,int ldc,float alpha){

    __m256 c0=_mm256_setzero_ps();
    __m256 c1=_mm256_setzero_ps();
    __m256 c2=_mm256_setzero_ps();
    __m256 c3=_mm256_setzero_ps();
    __m256 c4=_mm256_setzero_ps();
    __m256 c5=_mm256_setzero_ps();
    __m256 c6=_mm256_setzero_ps();
    __m256 c7=_mm256_setzero_ps();

    for(int k=0;k<kc;k++){

        __m256 b=_mm256_load_ps(&B[k*NR]);
        const float *a=&A[k*MR];

        c0=_mm256_fmadd_ps(_mm256_set1_ps(a[0]),b,c0);
        c1=_mm256_fmadd_ps(_mm256_set1_ps(a[1]),b,c1);
        c2=_mm256_fmadd_ps(_mm256_set1_ps(a[2]),b,c2);
        c3=_mm256_fmadd_ps(_mm256_set1_ps(a[3]),b,c3);
        c4=_mm256_fmadd_ps(_mm256_set1_ps(a[4]),b,c4);
        c5=_mm256_fmadd_ps(_mm256_set1_ps(a[5]),b,c5);
        c6=_mm256_fmadd_ps(_mm256_set1_ps(a[6]),b,c6);
        c7=_mm256_fmadd_ps(_mm256_set1_ps(a[7]),b,c7);
    }

    __m256 a=_mm256_set1_ps(alpha);

    _mm256_storeu_ps(&C[0*ldc],_mm256_add_ps(_mm256_loadu_ps(&C[0*ldc]),_mm256_mul_ps(a,c0)));
    _mm256_storeu_ps(&C[1*ldc],_mm256_add_ps(_mm256_loadu_ps(&C[1*ldc]),_mm256_mul_ps(a,c1)));
    _mm256_storeu_ps(&C[2*ldc],_mm256_add_ps(_mm256_loadu_ps(&C[2*ldc]),_mm256_mul_ps(a,c2)));
    _mm256_storeu_ps(&C[3*ldc],_mm256_add_ps(_mm256_loadu_ps(&C[3*ldc]),_mm256_mul_ps(a,c3)));
    _mm256_storeu_ps(&C[4*ldc],_mm256_add_ps(_mm256_loadu_ps(&C[4*ldc]),_mm256_mul_ps(a,c4)));
    _mm256_storeu_ps(&C[5*ldc],_mm256_add_ps(_mm256_loadu_ps(&C[5*ldc]),_mm256_mul_ps(a,c5)));
    _mm256_storeu_ps(&C[6*ldc],_mm256_add_ps(_mm256_loadu_ps(&C[6*ldc]),_mm256_mul_ps(a,c6)));
    _mm256_storeu_ps(&C[7*ldc],_mm256_add_ps(_mm256_loadu_ps(&C[7*ldc]),_mm256_mul_ps(a,c7)));
}

void syrk_fast(int n,int k,float alpha,const float* X,float beta,float* C,int threads){

#pragma omp parallel num_threads(threads)
{

#pragma omp for
for(long i=0;i<(long)n*n;i++)
    C[i]*=beta;

float *Ap=aligned_alloc_f(MC*KC);
float *Bp=aligned_alloc_f(KC*NC);

#pragma omp for collapse(2) schedule(dynamic,1)
for(int jc=0;jc<n;jc+=NC)
for(int ic=0;ic<n;ic+=MC)
{
    int nc=(jc+NC<n)?NC:n-jc;
    int mc=(ic+MC<n)?MC:n-ic;

    for(int pc=0;pc<k;pc+=KC)
    {
        int kc=(pc+KC<k)?KC:k-pc;

        pack_B(kc,nc,&X[jc+pc*n],n,Bp);
        pack_A(mc,kc,&X[ic+pc*n],n,Ap);

        for(int j=0;j<nc;j+=NR)
        for(int i=0;i<mc;i+=MR)
        {
            if(i+ic<j+jc) continue;

            kernel_8x8(kc,&Ap[i],&Bp[j],&C[(ic+i)+(jc+j)*n],n,alpha);
        }
    }
}

free(Ap);
free(Bp);

}
}

void testOurSyrkOnExample(){

    int n=3;
    int k=2;

    float A[6]={
        1,3,5,
        2,4,6
    };

    float C[9]={0};

    syrk_fast(n,k,1.0f,A,0.0f,C,1);

    printf("\nExample SYRK result:\n");

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            printf("%6.1f ",C[i+j*n]);
        printf("\n");
    }

    printf("\nExpected (lower triangle correct):\n");
    printf(" 5 11 17\n");
    printf("11 25 39\n");
    printf("17 39 61\n\n");
}

int main(){

testOurSyrkOnExample();

int n=20000;
int k=10000z;

float alpha=1.0f;
float beta=1.0f;

int thread_list[4]={1,2,4,8};

float *X=aligned_alloc_f((size_t)n*k);
float *C1=aligned_alloc_f((size_t)n*n);
float *C2=aligned_alloc_f((size_t)n*n);

for(long i=0;i<(long)n*k;i++)
    X[i]=(float)rand()/RAND_MAX;

printf("SYRK benchmark N=%d K=%d\n\n",n,k);

for(int t=0;t<4;t++){

int threads=thread_list[t];

memset(C1,0,(size_t)n*n*sizeof(float));
memset(C2,0,(size_t)n*n*sizeof(float));

printf("Threads %d\n",threads);

double t0=omp_get_wtime();
syrk_fast(n,k,alpha,X,beta,C1,threads);
double t1=omp_get_wtime()-t0;

openblas_set_num_threads(threads);

t0=omp_get_wtime();
cblas_ssyrk(CblasColMajor,CblasLower,CblasNoTrans,
            n,k,alpha,X,n,beta,C2,n);
double t2=omp_get_wtime()-t0;

printf("MyBlas      %.4f s\n",t1);
printf("OpenBLAS %.4f s\n",t2);
printf("Eff      %.2f %%\n\n",(t2/t1)*100.0);
}

free(X);
free(C1);
free(C2);

return 0;
}
