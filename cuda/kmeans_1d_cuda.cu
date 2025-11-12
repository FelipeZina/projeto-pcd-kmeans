/*
 * kmeans_1d_cuda.cu
 * K-means 1D - Implementação CUDA (Etapa 2)
 *
 * Compilar: nvcc -O2 kmeans_1d_cuda.cu -o kmeans_1d_cuda -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h> // Para CUDA

/* ---------- util CSV 1D: (Funções do professor) ---------- */
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

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
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
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); exit(1); }
        A[r] = atof(tok);
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
static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- KERNEL CUDA (Executa na GPU) ---------- */

// Kernel que calcula o assignment: 1 thread por ponto
__global__ void assignment_kernel(const double *d_X, const double *d_C, int *d_assign,
                                  int N, int K, double *d_sse_per_point)
{
    // Calcula o índice global desta thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Garante que a thread não ultrapasse o número de pontos
    if (idx < N) {
        double bestd = 1e300;
        int best_cluster = -1;

        // Cada thread (ponto) varre todos K centroides
        for (int c = 0; c < K; c++) {
            double diff = d_X[idx] - d_C[c];
            double d = diff * diff;
            if (d < bestd) {
                bestd = d;
                best_cluster = c;
            }
        }
        
        // Escreve o resultado na memória da GPU
        d_assign[idx] = best_cluster;
        d_sse_per_point[idx] = bestd; 
    }
}

/* ---------- GPU ASSIGNMENT (Função Host que gerencia a GPU) ---------- */
static double assignment_step_1d_gpu(const double *X, const double *C, int *assign, int N, int K)
{
    double *d_X, *d_C, *d_sse_per_point;
    int *d_assign;

    // --- Eventos CUDA para medição de tempo ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float time_H2D_ms = 0.0f;
    float time_Kernel_ms = 0.0f;
    float time_D2H_ms = 0.0f;
    // ------------------------------------------

    // 1. Alocar memória na GPU (Device)
    cudaMalloc((void**)&d_X, N * sizeof(double));
    cudaMalloc((void**)&d_C, K * sizeof(double));
    cudaMalloc((void**)&d_assign, N * sizeof(int));
    cudaMalloc((void**)&d_sse_per_point, N * sizeof(double));

    // 2. Copiar dados da CPU para a GPU (Host to Device - H2D)
    cudaEventRecord(start); // Inicia cronômetro
    cudaMemcpy(d_X, X, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, K * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop); // Para cronômetro
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_H2D_ms, start, stop); // Calcula tempo

    // 3. Configurar lançamento do Kernel
    int blockSize = 256; // <-- PONTO DE MEDIÇÃO! Mude para 128, 256, 512
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // 4. Lançar Kernel
    cudaEventRecord(start); // Inicia cronômetro
    assignment_kernel<<<gridSize, blockSize>>>(d_X, d_C, d_assign, N, K, d_sse_per_point);
    cudaEventRecord(stop); // Para cronômetro
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_Kernel_ms, start, stop); // Calcula tempo

    // 5. Copiar dados da GPU para a CPU (Device to Host - D2H)
    double *h_sse_per_point = (double*)malloc(N * sizeof(double));
    
    cudaEventRecord(start); // Inicia cronômetro
    cudaMemcpy(assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sse_per_point, d_sse_per_point, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); // Para cronômetro
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_D2H_ms, start, stop); // Calcula tempo

    // 6. Reduzir SSE no Host (CPU)
    double total_sse = 0.0;
    for (int i = 0; i < N; i++) {
        total_sse += h_sse_per_point[i];
    }
    
    // 7. Liberar memória
    cudaFree(d_X);
    cudaFree(d_C);
    cudaFree(d_assign);
    cudaFree(d_sse_per_point);
    free(h_sse_per_point);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Imprime os tempos desta iteração (para o relatório!)
    printf("    Tempos CUDA (iteração): H2D=%.2f ms, Kernel=%.2f ms, D2H=%.2f ms\n", 
           time_H2D_ms, time_Kernel_ms, time_D2H_ms);

    return total_sse;
}

/* ---------- UPDATE (Opção A: Roda na CPU) ---------- */
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K)
{
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }
    
    // Este laço roda na CPU
    for(int i=0; i<N; i++){
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else C[c] = X[0]; 
    }
    free(sum); free(cnt);
}

/* ---------- k-means 1D (Loop Principal) ---------- */
static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        printf("  Iteração %d...\n", it);
        // --- ETAPA DE ASSIGNMENT (GPU) ---
        sse = assignment_step_1d_gpu(X, C, assign, N, K);
        
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps && it > 0){ 
            printf("  Convergência alcançada.\n");
            break; 
        }
        
        // --- ETAPA DE UPDATE (CPU) ---
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [outAssign] [outCentroid]\n", argv[0]);
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); exit(1); }

    printf("Iniciando K-means 1D (CUDA)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);

    clock_t t0 = clock();
    int iters = 0; double sse = 0.0;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    clock_t t1 = clock();
    
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D (CUDA) finalizado.\n");
    printf("Iterações: %d | SSE final: %.6f | Tempo TOTAL: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);
    free(assign); free(X); free(C);
    return 0;
}
