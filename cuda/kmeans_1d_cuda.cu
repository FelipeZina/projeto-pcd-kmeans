/*
 * kmeans_1d_cuda.cu
 * K-means 1D - Implementação Híbrida (CUDA + OpenMP)
 * Etapa 2 - Opção A: Assignment na GPU, Update na CPU
 *
 * Convertido para 'float' para desempenho e compatibilidade com a GPU.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>        // Para FLT_MAX
#include <cuda_runtime.h> // Para CUDA
#include <omp.h>          // Para OpenMP no update

// --- MACRO DE VERIFICAÇÃO DE ERRO CUDA ---
// Checa erros em chamadas críticas da API CUDA
#define checkCuda(err) { \
  cudaError_t err_code = (err); \
  if (err_code != cudaSuccess) { \
    fprintf(stderr, "\n--- ERRO CUDA FATAL ---\n"); \
    fprintf(stderr, "Erro na linha %d: %s\n", __LINE__, cudaGetErrorString(err_code)); \
    fprintf(stderr, "Arquivo: %s\n", __FILE__); \
    exit(EXIT_FAILURE); \
  } \
}

/* ---------- Funções Utilitárias (Adaptadas para float) ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static float *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    float *A = (float*)malloc((size_t)R * sizeof(float));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }
    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = (float)atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}
static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}
static void write_centroids_csv(const char *path, const float *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/*
 * ===================================================================
 * ETAPA 1: ASSIGNMENT (Executa na GPU com CUDA)
 * ===================================================================
 */

__global__ void assignment_kernel(const float *d_X, const float *d_C, int *d_assign,
                                  int N, int K, float *d_sse_per_point)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float bestd = FLT_MAX;
        int best_cluster = -1;

        // Loop principal do kernel: 1 thread varre K centroides
        for (int c = 0; c < K; c++) {
            float diff = d_X[idx] - d_C[c];
            float d = diff * diff;
            if (d < bestd) {
                bestd = d;
                best_cluster = c;
            }
        }
        
        d_assign[idx] = best_cluster;
        d_sse_per_point[idx] = bestd; 
    }
}

// Função Host que gerencia a GPU e mede os tempos
static float assignment_step_1d_gpu(const float *X, const float *C, int *assign, 
                                    int N, int K, int blockSize,
                                    float *time_H2D_ms, float *time_Kernel_ms, float *time_D2H_ms)
{
    float *d_X, *d_C, *d_sse_per_point;
    int *d_assign;

    cudaEvent_t start, stop;
    checkCuda( cudaEventCreate(&start) );
    checkCuda( cudaEventCreate(&stop) );
    
    // 1. Alocar memória na GPU (Device)
    checkCuda( cudaMalloc((void**)&d_X, N * sizeof(float)) );
    checkCuda( cudaMalloc((void**)&d_C, K * sizeof(float)) );
    checkCuda( cudaMalloc((void**)&d_assign, N * sizeof(int)) );
    checkCuda( cudaMalloc((void**)&d_sse_per_point, N * sizeof(float)) );

    // 2. Copiar dados da CPU para a GPU (Host to Device - H2D)
    cudaEventRecord(start);
    checkCuda( cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_C, C, K * sizeof(float), cudaMemcpyHostToDevice) );
    cudaEventRecord(stop);
    checkCuda( cudaEventSynchronize(stop) );
    cudaEventElapsedTime(time_H2D_ms, start, stop);

    // 3. Configurar lançamento do Kernel
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // 4. Lançar Kernel
    cudaEventRecord(start);
    assignment_kernel<<<gridSize, blockSize>>>(d_X, d_C, d_assign, N, K, d_sse_per_point);
    checkCuda( cudaGetLastError() ); // Checa erro *após* o lançamento
    cudaEventRecord(stop);
    checkCuda( cudaEventSynchronize(stop) );
    cudaEventElapsedTime(time_Kernel_ms, start, stop);

    // 5. Copiar dados da GPU para a CPU (Device to Host - D2H)
    float *h_sse_per_point = (float*)malloc(N * sizeof(float));
    
    cudaEventRecord(start);
    checkCuda( cudaMemcpy(assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost) );
    checkCuda( cudaMemcpy(h_sse_per_point, d_sse_per_point, N * sizeof(float), cudaMemcpyDeviceToHost) );
    cudaEventRecord(stop);
    checkCuda( cudaEventSynchronize(stop) );
    cudaEventElapsedTime(time_D2H_ms, start, stop);

    // 6. Reduzir SSE no Host (CPU)
    double total_sse_double = 0.0; // Usar double para a soma previne erros de precisão
    for (int i = 0; i < N; i++) {
        total_sse_double += h_sse_per_point[i];
    }
    
    // 7. Liberar memória
    checkCuda( cudaFree(d_X) );
    checkCuda( cudaFree(d_C) );
    checkCuda( cudaFree(d_assign) );
    checkCuda( cudaFree(d_sse_per_point) );
    free(h_sse_per_point);
    checkCuda( cudaEventDestroy(start) );
    checkCuda( cudaEventDestroy(stop) );

    return (float)total_sse_double;
}

/*
 * ===================================================================
 * ETAPA 2: UPDATE (Executa na CPU com OpenMP)
 * ===================================================================
 */

// Função de update otimizada com OpenMP (da Etapa 1)
static void update_step_1d_omp(const float *X, float *C, const int *assign, int N, int K)
{
    // Usar 'double' para somas é mais seguro para evitar perda de precisão
    double *global_sum = (double*)calloc((size_t)K, sizeof(double));
    int    *global_cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!global_sum || !global_cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    #pragma omp parallel
    {
        double *local_sum = (double*)calloc((size_t)K, sizeof(double));
        int    *local_cnt = (int*)calloc((size_t)K, sizeof(int));

        #pragma omp for
        for(int i=0; i<N; i++){
            int a = assign[i];
            local_cnt[a] += 1;
            local_sum[a] += X[i]; // Soma 'float' em 'double'
        }

        #pragma omp critical
        {
            for(int c=0; c<K; c++){
                global_sum[c] += local_sum[c];
                global_cnt[c] += local_cnt[c];
            }
        }
        
        free(local_sum);
        free(local_cnt);
    }

    // Atualiza os centroides
    for(int c=0; c<K; c++){
        if(global_cnt[c] > 0) 
            C[c] = (float)(global_sum[c] / (double)global_cnt[c]); // Converte de volta para float
        else 
            C[c] = X[0];
    }
    
    free(global_sum);
    free(global_cnt);
}

/*
 * ===================================================================
 * Loop Principal e Main
 * ===================================================================
 */
static void kmeans_1d(const float *X, float *C, int *assign,
                      int N, int K, int max_iter, float eps, int blockSize,
                      int *iters_out, float *sse_out)
{
    float prev_sse = FLT_MAX;
    float sse = 0.0f;
    
    float total_H2D_ms = 0.0f;
    float total_Kernel_ms = 0.0f;
    float total_D2H_ms = 0.0f;
    
    int it;
    for(it=0; it<max_iter; it++){
        float t_h2d=0, t_kern=0, t_d2h=0;
        
        // --- ETAPA 1: ASSIGNMENT (GPU) ---
        sse = assignment_step_1d_gpu(X, C, assign, N, K, blockSize, 
                                     &t_h2d, &t_kern, &t_d2h);
        
        total_H2D_ms += t_h2d;
        total_Kernel_ms += t_kern;
        total_D2H_ms += t_d2h;
        
        float rel = fabsf(sse - prev_sse) / (prev_sse > 0.0f ? prev_sse : 1.0f);

        // A lógica de parada agora é silenciosa, sem printf
        if(rel < eps && it > 0){ 
            it++; 
            break; 
        }
        
        // --- ETAPA 2: UPDATE (CPU + OpenMP) ---
        update_step_1d_omp(X, C, assign, N, K);
        prev_sse = sse;
    }
    
    *iters_out = it;
    *sse_out = sse;

    // Imprime o total dos tempos de GPU (APENAS NO FINAL)
    printf("\n  --- Tempos Totais (GPU) ---\n");
    printf("  Total H2D:     %.2f ms\n", total_H2D_ms);
    printf("  Total Kernel:  %.2f ms\n", total_Kernel_ms);
    printf("  Total D2H:     %.2f ms\n", total_D2H_ms);
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [blockSize=256] [max_iter=50] [eps=1e-4] [outAssign] [outCentroid]\n", argv[0]);
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    
    int blockSize = (argc>3)? atoi(argv[3]) : 256; 
    int max_iter  = (argc>4)? atoi(argv[4]) : 50;
    float eps     = (argc>5)? (float)atof(argv[5]) : 1e-4f;
    const char *outAssign   = (argc>6)? argv[6] : NULL;
    const char *outCentroid = (argc>7)? argv[7] : NULL;

    if(blockSize <= 0 || max_iter <= 0 || eps <= 0.0f){
        fprintf(stderr,"Parâmetros inválidos: blockSize>0, max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    float *X = read_csv_1col(pathX, &N);
    float *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); exit(1); }

    printf("Iniciando K-means 1D (CUDA+OpenMP Híbrido)\n");
    printf("N=%d K=%d blockSize=%d max_iter=%d eps=%g\n", N, K, blockSize, max_iter, eps);

    // Medição de tempo principal (Tempo TOTAL)
    clock_t t0 = clock();
    int iters = 0; float sse = 0.0f;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, blockSize, &iters, &sse);
    clock_t t1 = clock();
    
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    // Impressão final dos resultados
    printf("\nK-means 1D (CUDA+OpenMP) finalizado.\n");
    printf("Iterações: %d | SSE final: %.6f | Tempo TOTAL: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);
    free(assign); free(X); free(C);
    return 0;
}
