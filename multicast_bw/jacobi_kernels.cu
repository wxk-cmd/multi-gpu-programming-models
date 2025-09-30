/* Multicast-only kernels for Hopper (SM90+) — no reduction, no sum.
 * Writes via multimem.st so one store is replicated to all GPUs bound
 * to the same multicast object.
 */
#include <cassert>
#include <cstdio>
#include <cuda/atomic>

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t _st = call;                                                             \
        if (_st != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA RT error %s at %s:%d (%d)\n",                             \
                    cudaGetErrorString(_st), __FILE__, __LINE__, _st);                      \
            exit(_st);                                                                      \
        }                                                                                   \
    }

#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

//====================================================
// 1) 到达计数 + 栅栏（与之前一致；累计计数法）
//    counter_mc / counter_uc 指向“同一物理内存”的两份映射
//====================================================
__global__ void mc_barrier_kernel(unsigned int* counter_uc, // UC mapping
                                  unsigned int* counter_mc, // MC mapping
                                  unsigned int  expected_count)
{
    assert(gridDim.x * blockDim.x == 1);
#if __CUDA_ARCH__ >= 900
    // 向所有副本原子加 1（release）
    asm volatile ("multimem.red.release.sys.global.add.u32 [%0], %1;"
                  : : "l"(counter_mc), "n"(1) : "memory");

    // alias fence：建立不同代理映射间的顺序
    asm volatile ("fence.proxy.alias;" ::: "memory");

    // 在 UC 上自旋等待（acquire）
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> ac(*counter_uc);
    while (ac.load(cuda::memory_order_acquire) < expected_count) { /* spin */ }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("ERROR: mc_barrier_kernel needs __CUDA_ARCH__ >= 900\n");
    }
#endif
}

//====================================================
// 2) 多播写 kernel：把本地 in 写到 MC 数组的“我的槽位”
//    布局：mc_base[ slot_idx * N + i ]
//    注意：multimem.st 写入会被复制到所有 GPU 的该地址
//====================================================
__global__ void mc_multicast_store_kernel(const real* __restrict__ in,
                                          real* __restrict__ mc_base, // MC mapping
                                          size_t N,
                                          size_t slot_idx)
{
#if __CUDA_ARCH__ >= 900
    const size_t base = slot_idx * N;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += (size_t)blockDim.x * gridDim.x)
    {
#ifdef USE_DOUBLE
        double v = (double)in[i];
        asm volatile ("multimem.st.release.sys.global.f64 [%0], %1;"
                      :
                      : "l"(mc_base + base + i), "d"(v)
                      : "memory");
#else
        float v = (float)in[i];
        asm volatile ("multimem.st.release.sys.global.f32 [%0], %1;"
                      :
                      : "l"(mc_base + base + i), "f"(v)
                      : "memory");
#endif
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("ERROR: mc_multicast_store_kernel needs __CUDA_ARCH__ >= 900\n");
    }
#endif
}

//====================================================
// 3) （可选）从 UC 映射把“所有槽位”搬到本地 out（纯本地读）
//    仅用于验证/聚合，不涉及多播指令
//====================================================
__global__ void gather_from_uc_slots_kernel(const real* __restrict__ uc_base,
                                            real* __restrict__ out, // size = num_gpus*N
                                            size_t N, int num_gpus)
{
    size_t total = (size_t)num_gpus * N;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (size_t)blockDim.x * gridDim.x)
    {
        out[i] = uc_base[i];
    }
}

//====================================================
// 4) Launcher：一站式多播通信（不做求和）
//    - 每个 GPU 将本地 in 多播写入其槽位
//    - 做一次“累计到达”栅栏，确保所有写完成
//    - 如需把所有槽位搬回本地缓冲，设置 do_gather=true
//====================================================
void launch_mc_multicast_1d(const real* local_in,  // [N]
                            real* mc_array,        // MC 映射基址，大小 num_gpus*N
                            real* uc_array,        // UC 映射基址，大小同上
                            unsigned int* counter_mc,
                            unsigned int* counter_uc,
                            size_t N,
                            int num_gpus,
                            int my_slot,           // 本 GPU 的槽位索引（通常等于本 rank）
                            int iter,              // 第几轮（从0开始），用于 expected_count
                            bool do_gather,        // 是否把所有槽位读回本地 out_all
                            real* out_all,         // [num_gpus*N]（仅 do_gather=true 时使用）
                            cudaStream_t stream)
{
    if (N == 0) return;

    const int threads = 256;
    const int blocks  = (int)std::min<size_t>((N + threads - 1) / threads, 65535);

    // (1) 多播写：把 in -> MC 槽位（一次写，所有 GPU 同步收到）
    mc_multicast_store_kernel<<<blocks, threads, 0, stream>>>(
        local_in, mc_array, N, (size_t)my_slot);
    CUDA_RT_CALL(cudaGetLastError());

    // (2) 累计到达栅栏：expected = num_gpus * (iter+1)
    unsigned int expected = (unsigned int)num_gpus;
    expected *= (unsigned int)(iter + 1);
    mc_barrier_kernel<<<1, 1, 0, stream>>>(counter_uc, counter_mc, expected);
    CUDA_RT_CALL(cudaGetLastError());

    // (3) 可选：从 UC 读回所有槽位（纯本地操作）
    if (do_gather && out_all) {
        const size_t total = (size_t)num_gpus * N;
        const int blocks2 = (int)std::min<size_t>((total + threads - 1) / threads, 65535);
        gather_from_uc_slots_kernel<<<blocks2, threads, 0, stream>>>(
            uc_array, out_all, N, num_gpus);
        CUDA_RT_CALL(cudaGetLastError());
    }
}

//====================================================
// 5) 兼容旧符号（如果主程序还在链接这几个名字）
//====================================================
extern "C" {
void launch_initialize_boundaries(real*, real*, const real, const int, const int, const int, const int) {}
void launch_jacobi_kernel(real*, const real*, real*, const int, const int, const int, const bool, cudaStream_t) {}
void launch_jacobi_p2p_kernel(real*, const real*, real*, const int, const int, const int,
                              real*, const int, real*, const int, const bool, cudaStream_t) {}
} // extern "C"
