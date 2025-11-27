/*
 * kmeans_1d_mpi.c
 * K-means 1D - Implementação Distribuída (MPI)
 * Etapa 3
 *
 * Compilar: mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
 * Executar: mpirun -np 4 --allow-run-as-root ./kmeans_1d_mpi ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h> // Biblioteca MPI

/* --- Funções de Leitura de Arquivo (Apenas Rank 0 usa) --- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ return 0; }
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
    if(R<=0){ return NULL; }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    FILE *f = fopen(path, "r");
    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        char *tok = strtok(line, ",; \t");
        if(tok){
            A[r++] = atof(tok);
            if(r >= R) break;
        }
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f) return;
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* --- Main MPI --- */
int main(int argc, char **argv){
    // 1. Inicialização do MPI
    MPI_Init(&argc, &argv);

    int rank, n_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Quem sou eu?
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs); // Quantos somos?

    // Parâmetros principais
    int N_global = 0, K = 0;
    int max_iter = 50;
    double eps = 1e-4;
    
    // Arrays globais (apenas Rank 0 aloca/lê estes)
    double *X_global = NULL;
    double *C = NULL; // Todos terão C, mas Rank 0 lê o inicial

    // --- LEITURA DE DADOS (Apenas Rank 0) ---
    if(rank == 0){
        if(argc < 3){
            fprintf(stderr, "Uso: mpirun ... %s dados.csv centroides.csv [max_iter] [eps]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const char *pathX = argv[1];
        const char *pathC = argv[2];
        if(argc > 3) max_iter = atoi(argv[3]);
        if(argc > 4) eps = atof(argv[4]);

        X_global = read_csv_1col(pathX, &N_global);
        C = read_csv_1col(pathC, &K);

        if(!X_global || !C){
            fprintf(stderr, "Erro ao ler arquivos no Rank 0.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("MPI Iniciado com %d processos.\n", n_procs);
        printf("Dados lidos: N=%d, K=%d\n", N_global, K);
    }

    // 2. Broadcast de Dimensões (N, K, max_iter, eps)
    // Rank 0 avisa todo mundo qual o tamanho do problema
    MPI_Bcast(&N_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Se não sou rank 0, aloco espaço para C (pois vou receber no Bcast depois)
    if(rank != 0) {
        C = (double*)malloc(K * sizeof(double));
    }

    // --- DISTRIBUIÇÃO DOS PONTOS (SCATTER) ---
    // Precisamos dividir N_global pontos entre n_procs processos.
    // Usamos MPI_Scatterv para lidar com divisões não exatas.
    
    int *sendcounts = (int*)malloc(n_procs * sizeof(int));
    int *displs = (int*)malloc(n_procs * sizeof(int));
    
    // Calcula quantos pontos cada processo recebe
    int rem = N_global % n_procs;
    int sum = 0;
    for(int i = 0; i < n_procs; i++) {
        sendcounts[i] = N_global / n_procs;
        if(i < rem) sendcounts[i]++; // Distribui o resto
        displs[i] = sum;
        sum += sendcounts[i];
    }

    // Número de pontos locais deste processo específico
    int N_local = sendcounts[rank];
    double *X_local = (double*)malloc(N_local * sizeof(double));

    // Espalha os dados! (Rank 0 envia pedaços de X_global para X_local de cada um)
    MPI_Scatterv(X_global, sendcounts, displs, MPI_DOUBLE, 
                 X_local, N_local, MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);

    // Arrays auxiliares locais para o cálculo
    int *assign_local = (int*)malloc(N_local * sizeof(int));
    double *sum_local = (double*)malloc(K * sizeof(double));
    int *cnt_local = (int*)malloc(K * sizeof(int));
    
    // Arrays globais para redução (usados para atualizar C)
    double *sum_global = (double*)malloc(K * sizeof(double));
    int *cnt_global = (int*)malloc(K * sizeof(int));

    // Medição de tempo
    double comm_time_total = 0.0;
    double start_time = MPI_Wtime();

    double prev_sse = 1e300;
    double sse_global = 0.0;
    int it = 0;

    // --- LOOP DO K-MEANS ---
    for(it = 0; it < max_iter; it++){
        
        // 1. Broadcast dos Centroides (Passo 1 do enunciado)
        // Garante que todos tenham o C atualizado antes de começar
        // (Embora Allreduce no final já atualize, o enunciado pede explicitamente)
        MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // 2. Assignment Local (Passo 2 do enunciado)
        double sse_local = 0.0;
        
        // Zera acumuladores locais
        for(int c=0; c<K; c++) { sum_local[c] = 0.0; cnt_local[c] = 0; }

        for(int i=0; i<N_local; i++){
            double best_dist = 1e300;
            int best_cluster = -1;
            
            // Encontra cluster mais próximo
            for(int c=0; c<K; c++){
                double diff = X_local[i] - C[c];
                double d = diff*diff;
                if(d < best_dist){
                    best_dist = d;
                    best_cluster = c;
                }
            }
            
            assign_local[i] = best_cluster;
            sse_local += best_dist;
            
            // Acumula para o Update (Opção A do enunciado OpenMP adaptada para MPI)
            sum_local[best_cluster] += X_local[i];
            cnt_local[best_cluster]++;
        }

        // 3. Redução Global (Passo 3 do enunciado)
        double t_comm_start = MPI_Wtime();

        // A) Soma SSE_local -> SSE_global (Reduce para Rank 0 verificar convergência)
        MPI_Reduce(&sse_local, &sse_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // B) Soma sum_local e cnt_local de TODOS -> TODOS (Allreduce)
        // Isso permite que TODOS atualizem o C simultaneamente
        MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        comm_time_total += (MPI_Wtime() - t_comm_start);

        // C) Atualizar C (Cada processo atualiza C com resultados globais)
        for(int c=0; c<K; c++){
            if(cnt_global[c] > 0)
                C[c] = sum_global[c] / cnt_global[c];
            else
                C[c] = (rank == 0) ? X_global[0] : 0.0; // Tratamento simplificado
        }

        // 4. Verificação de Convergência (Controlada pelo Rank 0)
        int stop_flag = 0;
        if(rank == 0){
            double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
            printf("  [Iter %d] SSE Global: %.6f\n", it, sse_global);
            if(rel < eps) stop_flag = 1;
            prev_sse = sse_global;
        }
        
        // Avisa a todos se deve parar ou continuar
        MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(stop_flag) {
            if(rank==0) it++; // conta a última
            break;
        }
    }

    double end_time = MPI_Wtime();
    double total_time = (end_time - start_time) * 1000.0; // ms
    comm_time_total *= 1000.0; // ms

    // --- RESULTADOS FINAIS ---
    if(rank == 0){
        printf("\nK-means 1D (MPI) Finalizado.\n");
        printf("N=%d, K=%d, Procs=%d\n", N_global, K, n_procs);
        printf("Iterações: %d | SSE Final: %.6f\n", it, sse_global);
        printf("Tempo TOTAL: %.2f ms\n", total_time);
        printf("Tempo COMUNICAÇÃO (Reduce/Allreduce): %.2f ms\n", comm_time_total);
        
        // Opcional: Salvar CSV final (Rank 0 tem o C final, mas não tem o assign global completo aqui para salvar fácil)
        // Para salvar 'assign.csv' completo, precisaríamos de MPI_Gatherv, mas o enunciado foca em métricas.
        // Vamos salvar apenas os centroides.
        if(argc > 6) write_centroids_csv(argv[6], C, K);
    }

    // Limpeza
    free(X_local); free(C); free(assign_local);
    free(sum_local); free(cnt_local);
    free(sum_global); free(cnt_global);
    free(sendcounts); free(displs);
    if(rank == 0) free(X_global);

    MPI_Finalize();
    return 0;
}
