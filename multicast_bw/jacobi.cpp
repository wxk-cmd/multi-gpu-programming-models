// Test for multicast-only 1D communication (no reduction).
// Builds a multicast object for a slot-layout buffer: num_gpus * N elements.
// Each GPU writes its slot via multimem.st.* and uses a cumulative-arrival counter
// for a per-iteration barrier. Optional gather from UC for verification.
//
// Compile example (Hopper / SM90+):
//   nvcc -O3 -std=c++17 -arch=sm_90 -c mc_multicast_only_kernels.cu -o mc_multicast_only_kernels.o
//   mpicxx -O3 -std=c++17 test_mc_multicast_only.cpp mc_multicast_only_kernels.o \
//     -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcuda -lcudart -o test_mc_mc
//
// Run example:
//   mpirun -np 4 ./test_mc_mc -N 134217728 -iters 200 -warmup 20
//
// Optional flags:
//   -csv         : print csv line
//   -gather      : after each iter, gather all slots from UC to a local buffer (validation cost)
//   -dtype double: print dtype (real type still decided by USE_DOUBLE macro at build)

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MPI_CALL(call)                                                                \
    do {                                                                              \
        int _s = (call);                                                              \
        if (_s != MPI_SUCCESS) {                                                      \
            char errstr[MPI_MAX_ERROR_STRING]; int len=0;                             \
            MPI_Error_string(_s, errstr, &len);                                       \
            fprintf(stderr, "MPI error %s at %s:%d (%d)\n",                           \
                    (len?errstr:"<unknown>"), __FILE__, __LINE__, _s);                \
            exit(_s);                                                                 \
        }                                                                             \
    } while(0)

#define CUDA_RT_CALL(call)                                                                  \
    do {                                                                                    \
        cudaError_t _st = (call);                                                           \
        if (_st != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA RT error %s at %s:%d (%d)\n",                             \
                    cudaGetErrorString(_st), __FILE__, __LINE__, _st);                      \
            exit(_st);                                                                      \
        }                                                                                   \
    } while(0)

#define CUDA_CALL(call)                                                            \
    do {                                                                           \
        CUresult _st = (call);                                                     \
        if (_st != CUDA_SUCCESS) {                                                 \
            const char* es = nullptr;                                              \
            cuGetErrorString(_st, &es);                                            \
            fprintf(stderr, "CUDA Driver error %s at %s:%d (%d)\n",                \
                    (es?es:"<unknown>"), __FILE__, __LINE__, _st);                 \
            exit(_st);                                                             \
        }                                                                          \
    } while(0)

#ifdef USE_DOUBLE
using real = double;
#else
using real = float;
#endif

// ---------- helpers ----------
template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T defval) {
    T v = defval;
    auto it = std::find(begin, end, arg);
    if (it != end && ++it != end) { std::istringstream(*it) >> v; }
    return v;
}
inline bool get_arg(char** begin, char** end, const std::string& arg) {
    return std::find(begin, end, arg) != end;
}
template <typename T>
T round_up(T x, T g) { return (x + g - 1) & ~(g - 1); }

// ---------- external kernel launcher (from mc_multicast_only_kernels.cu) ----------
extern void launch_mc_multicast_1d(const real* local_in,
                                   real* mc_array,
                                   real* uc_array,
                                   unsigned int* counter_mc,
                                   unsigned int* counter_uc,
                                   size_t N,
                                   int num_gpus,
                                   int my_slot,
                                   int iter,
                                   bool do_gather,
                                   real* out_all,
                                   cudaStream_t stream);

// ---------- main ----------
int main(int argc, char** argv) {
    MPI_CALL(MPI_Init(&argc, &argv));
    int rank=0, world=1;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &world));

    int dev_count = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&dev_count));

    // split to local comm to get local_rank for device selection
    MPI_Comm local_comm; MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank=0, local_size=1;
    MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CALL(MPI_Comm_size(local_comm, &local_size));
    MPI_CALL(MPI_Comm_free(&local_comm));

    if (dev_count <= 0) {
        if (rank==0) fprintf(stderr, "No CUDA device.\n");
        MPI_CALL(MPI_Finalize());
        return 1;
    }
    int dev_id = local_rank % dev_count;
    CUDA_RT_CALL(cudaSetDevice(dev_id));
    CUDA_RT_CALL(cudaFree(0));

    // args
    const size_t N       = get_argval<size_t>(argv, argv+argc, std::string("-N"),      size_t(1)<<30); // ~67M elems default
    const int    iters   = get_argval<int>(argv, argv+argc, std::string("-iters"),     10);
    const int    warmup  = get_argval<int>(argv, argv+argc, std::string("-warmup"),    5);
    const bool   csv     = get_arg(argv, argv+argc, std::string("-csv"));
    const bool   doGather= get_arg(argv, argv+argc, std::string("-gather"));
    const std::string dtype = get_argval<std::string>(argv, argv+argc, std::string("-dtype"),
#ifdef USE_DOUBLE
                                                      "double");
#else
                                                      "float");
#endif

    // Check FABRIC & MULTICAST
    int fabric_supp=0, mc_supp=0;
    CUDA_CALL(cuInit(0));
    CUDA_CALL(cuDeviceGetAttribute(&fabric_supp, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev_id));
    CUDA_CALL(cuDeviceGetAttribute(&mc_supp,     CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,          dev_id));
    if (!fabric_supp || !mc_supp) {
        cudaDeviceProp prop; CUDA_RT_CALL(cudaGetDeviceProperties(&prop, dev_id));
        fprintf(stderr, "Rank %d dev %d (%s) does not support FABRIC(%d) or MULTICAST(%d)\n",
                rank, dev_id, prop.name, fabric_supp, mc_supp);
        MPI_CALL(MPI_Finalize()); return 1;
    }
    if (dev_count > 1 && dev_count < local_size) {
        // 多进程复用同一 GPU (MPS oversubscribe) 仍可使用多播，但注意性能
        if (rank==0) fprintf(stderr, "Warning: %d ranks sharing %d GPUs on this node.\n", local_size, dev_count);
    }

    // ---------- build MC object for slots buffer ----------
    // Total elements = world * N
    const size_t elem_mc = (size_t)world * N;
    const size_t bytes_mc = elem_mc * sizeof(real);

    CUmulticastObjectProp mc_prop = {};
    mc_prop.numDevices  = world;
    mc_prop.size        = bytes_mc;
    mc_prop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    size_t gran_min=0, gran_rec=0;
    CUDA_CALL(cuMulticastGetGranularity(&gran_min, &mc_prop, CU_MULTICAST_GRANULARITY_MINIMUM));
    CUDA_CALL(cuMulticastGetGranularity(&gran_rec, &mc_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    mc_prop.size = round_up(mc_prop.size, gran_rec); // recommended alignment

    // UC allocation size must respect minimum granularity
    const size_t uc_size = round_up(bytes_mc, gran_min);

    CUmemGenericAllocationHandle uc_handle_slots{};
    // Create UC physical memory on this device
    {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = dev_id;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        CUDA_CALL(cuMemCreate(&uc_handle_slots, uc_size, &prop, 0));
    }

    // Create / share multicast object
    CUmemGenericAllocationHandle mc_handle_slots{};
    CUmemFabricHandle mc_fh_slots{};
    if (rank == 0) {
        CUDA_CALL(cuMulticastCreate(&mc_handle_slots, &mc_prop));
        CUDA_CALL(cuMemExportToShareableHandle(&mc_fh_slots, mc_handle_slots, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    MPI_CALL(MPI_Bcast(&mc_fh_slots, sizeof(CUmemFabricHandle), MPI_BYTE, 0, MPI_COMM_WORLD));
    if (rank != 0) {
        CUDA_CALL(cuMemImportFromShareableHandle(&mc_handle_slots, &mc_fh_slots, CU_MEM_HANDLE_TYPE_FABRIC));
    }

    // Add this device to multicast
    CUDA_CALL(cuMulticastAddDevice(mc_handle_slots, dev_id));
    // Ensure all devices are added before binding memory on any device
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

    // Bind our UC physical mem into multicast object at offset 0
    CUDA_CALL(cuMulticastBindMem(mc_handle_slots, 0 /*mc_off*/, uc_handle_slots, 0 /*uc_off*/, uc_size, 0));

    // Map MC & UC to VA
    CUdeviceptr mc_ptr_slots=0, uc_ptr_slots=0;
    {
        // MC
        CUDA_CALL(cuMemAddressReserve(&mc_ptr_slots, mc_prop.size, gran_rec, 0, 0));
        CUDA_CALL(cuMemMap(mc_ptr_slots, mc_prop.size, 0, mc_handle_slots, 0));
        CUmemAccessDesc ad{}; ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id = dev_id; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUDA_CALL(cuMemSetAccess(mc_ptr_slots, mc_prop.size, &ad, 1));
        // UC
        CUDA_CALL(cuMemAddressReserve(&uc_ptr_slots, uc_size, gran_min, 0, 0));
        CUDA_CALL(cuMemMap(uc_ptr_slots, uc_size, 0, uc_handle_slots, 0));
        CUDA_CALL(cuMemSetAccess(uc_ptr_slots, uc_size, &ad, 1));
    }
    real* mc_slots = reinterpret_cast<real*>(mc_ptr_slots);
    real* uc_slots = reinterpret_cast<real*>(uc_ptr_slots);

    // ---------- build MC/UC for 4B cumulative counter ----------
    const size_t ctr_bytes = round_up<size_t>(sizeof(unsigned int), gran_min);
    CUmemGenericAllocationHandle uc_handle_ctr{};
    {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = dev_id;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        CUDA_CALL(cuMemCreate(&uc_handle_ctr, ctr_bytes, &prop, 0));
    }

    CUmemGenericAllocationHandle mc_handle_ctr{};
    CUmemFabricHandle mc_fh_ctr{};
    if (rank == 0) {
        CUmulticastObjectProp cprop{};
        cprop.numDevices  = world;
        cprop.size        = ctr_bytes;
        cprop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        CUDA_CALL(cuMulticastCreate(&mc_handle_ctr, &cprop));
        CUDA_CALL(cuMemExportToShareableHandle(&mc_fh_ctr, mc_handle_ctr, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    MPI_CALL(MPI_Bcast(&mc_fh_ctr, sizeof(CUmemFabricHandle), MPI_BYTE, 0, MPI_COMM_WORLD));
    if (rank != 0) {
        CUDA_CALL(cuMemImportFromShareableHandle(&mc_handle_ctr, &mc_fh_ctr, CU_MEM_HANDLE_TYPE_FABRIC));
    }
    CUDA_CALL(cuMulticastAddDevice(mc_handle_ctr, dev_id));
    printf("rank:%d,cuMulticastAddDevice:%d",rank,dev_id);
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CALL(cuMulticastBindMem(mc_handle_ctr, 0, uc_handle_ctr, 0, ctr_bytes, 0));

    CUdeviceptr mc_ptr_ctr=0, uc_ptr_ctr=0;
    {
        CUDA_CALL(cuMemAddressReserve(&mc_ptr_ctr, ctr_bytes, gran_min, 0, 0));
        CUDA_CALL(cuMemMap(mc_ptr_ctr, ctr_bytes, 0, mc_handle_ctr, 0));
        CUmemAccessDesc ad{}; ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id = dev_id; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUDA_CALL(cuMemSetAccess(mc_ptr_ctr, ctr_bytes, &ad, 1));

        CUDA_CALL(cuMemAddressReserve(&uc_ptr_ctr, ctr_bytes, gran_min, 0, 0));
        CUDA_CALL(cuMemMap(uc_ptr_ctr, ctr_bytes, 0, uc_handle_ctr, 0));
        CUDA_CALL(cuMemSetAccess(uc_ptr_ctr, ctr_bytes, &ad, 1));
    }
    auto* counter_mc = reinterpret_cast<unsigned int*>(mc_ptr_ctr);
    auto* counter_uc = reinterpret_cast<unsigned int*>(uc_ptr_ctr);

    // init counter=0
    CUDA_RT_CALL(cudaMemset(counter_uc, 0, sizeof(unsigned int)));

    // ---------- local buffers ----------
    real* d_in = nullptr;
    CUDA_RT_CALL(cudaMalloc(&d_in, N * sizeof(real)));
    // Initialize input with deterministic values
    // (just fill with rank-dependent pattern)
    {
        std::vector<real> h(N);
        const real base = real(rank + 1);
        for (size_t i=0;i<N;i++) h[i] = base + real(i % 17) * real(0.001);
        CUDA_RT_CALL(cudaMemcpy(d_in, h.data(), N*sizeof(real), cudaMemcpyHostToDevice));
    }

    // Optional gather buffer (to read back all slots from UC)
    real* d_all = nullptr;
    if (doGather) {
        CUDA_RT_CALL(cudaMalloc(&d_all, (size_t)world * N * sizeof(real)));
    }

    // stream
    cudaStream_t stream; CUDA_RT_CALL(cudaStreamCreate(&stream));

    // ---------- warmup ----------
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    for (int w=0; w<warmup; ++w) {
        CUDA_RT_CALL(cudaMemcpyAsync(mc_slots + rank * N,  // dst
                        d_in,                // src
                        N * sizeof(real),    // size
                        cudaMemcpyDeviceToDevice,
                        stream));   
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    }
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

    // ---------- timed iters ----------
    const double t0 = MPI_Wtime();
    for (int it=0; it<iters; ++it) {
        CUDA_RT_CALL(cudaMemcpyAsync(mc_slots + rank * N,  // dst
                        d_in,                // src
                        N * sizeof(real),    // size
                        cudaMemcpyDeviceToDevice,
                        stream));
    	CUDA_RT_CALL(cudaStreamSynchronize(stream));
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    const double t1 = MPI_Wtime();

    // ---------- bandwidth ----------
    const double secs = t1 - t0;
    const double bytes_per_iter_per_rank = double(N) * double(sizeof(real)); // one multicast store per element
    const double total_bytes_per_rank    = bytes_per_iter_per_rank * double(iters);
    const double gb_per_rank             = total_bytes_per_rank / 1.0e9;
    const double gbps_per_rank           = secs > 0 ? (gb_per_rank / secs) : 0.0;

    // 可选：计算“交付总字节”（硬件复制到 world 份）
    const double delivered_gb_per_rank   = gb_per_rank * double(world);
    const double delivered_gbps_per_rank = secs > 0 ? (delivered_gb_per_rank / secs) : 0.0;

    double avg_gbps=0.0, sum_gbps=0.0;
    MPI_CALL(MPI_Allreduce(&gbps_per_rank, &sum_gbps, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    avg_gbps = sum_gbps / double(world);

    if (csv) {
        if (rank==0) {
            // name, world, dev_count_node, N, iters, warmup, dtype, gbps_per_rank_avg, gbps_sum, secs
            printf("mc_multicast_only,%d,%d,%zu,%d,%d,%s,%.6f,%.6f,%.6f\n",
                   world, dev_count, N, iters, warmup,
#ifdef USE_DOUBLE
                   "double",
#else
                   "float",
#endif
                   avg_gbps, sum_gbps, secs);
            fflush(stdout);
        }
    } else {
        printf("[Rank %d | Dev %d] N=%zu | per-rank write: %.3f GB in %.3f s => %.3f GB/s\n",
               rank, dev_id, N, gb_per_rank, secs, gbps_per_rank);
        printf("[Rank %d] delivered (replicated to %d GPUs): %.3f GB/s\n",
               rank, world, delivered_gbps_per_rank);
        if (rank==0) {
            printf("World=%d | Avg per-rank BW: %.3f GB/s | Aggregate: %.3f GB/s | dtype=%s\n",
                   world, avg_gbps, sum_gbps,
#ifdef USE_DOUBLE
                   "double"
#else
                   "float"
#endif
            );
        }
    }

    // ---------- cleanup ----------
    CUDA_RT_CALL(cudaStreamDestroy(stream));
    if (d_all) CUDA_RT_CALL(cudaFree(d_all));
    CUDA_RT_CALL(cudaFree(d_in));

    // unmap & release counter
    CUDA_CALL(cuMemUnmap(mc_ptr_ctr, ctr_bytes));
    CUDA_CALL(cuMemUnmap(uc_ptr_ctr, ctr_bytes));
    CUDA_CALL(cuMemRelease(uc_handle_ctr));
    CUDA_CALL(cuMemRelease(mc_handle_ctr));
    CUDA_CALL(cuMemAddressFree(mc_ptr_ctr, ctr_bytes));
    CUDA_CALL(cuMemAddressFree(uc_ptr_ctr, ctr_bytes));

    // unmap & release slots
    CUDA_CALL(cuMemUnmap(mc_ptr_slots, mc_prop.size));
    CUDA_CALL(cuMemUnmap(uc_ptr_slots, uc_size));
    CUDA_CALL(cuMemRelease(uc_handle_slots));
    CUDA_CALL(cuMemRelease(mc_handle_slots));
    CUDA_CALL(cuMemAddressFree(mc_ptr_slots, mc_prop.size));
    CUDA_CALL(cuMemAddressFree(uc_ptr_slots, uc_size));

    MPI_CALL(MPI_Finalize());
    return 0;
}
