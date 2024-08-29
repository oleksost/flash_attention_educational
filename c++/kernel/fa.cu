#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void flash_attn_forward_kernel(
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    float *__restrict__ out,
    int T_r,
    int T_c,
    float tau,
    float *l_hbm,
    float *m_hbm,
    int b,
    int h,
    int s,
    int d_head,
    int B_r,
    int B_c)
{
    int block_b = blockIdx.x;   // batch index
    int block_h = blockIdx.y;   // head index
    int thread_i = threadIdx.x; // thread index within the block

    extern __shared__ float shared_memory[]; // Dynamic shared memory
    float *K_shared = shared_memory;
    float *V_shared = &shared_memory[B_c * d_head];
    float *Q_shared = &shared_memory[(B_c + B_c) * d_head];

    for (int j = 0; j < T_r; j++)
    {
        int r_idx = j * B_r + thread_i;
        if (r_idx < s)
        {
            for (int c = 0; c < d_head; c++)
            {
                Q_shared[thread_i * d_head + c] = q[block_b * h * s * d_head + block_h * s * d_head + r_idx * d_head + c];
            }
        }
        __syncthreads();

        float l = 0.0f, m = 0.0f;
        for (int i = 0; i < T_c; i++)
        {
            int c_idx = i * B_c + thread_i;
            if (c_idx < s)
            {
                for (int c = 0; c < d_head; c++)
                {
                    K_shared[thread_i * d_head + c] = k[block_b * h * s * d_head + block_h * s * d_head + c_idx * d_head + c];
                    V_shared[thread_i * d_head + c] = v[block_b * h * s * d_head + block_h * s * d_head + c_idx * d_head + c];
                }
            }
            __syncthreads();

            if (r_idx < s)
            {
                for (int b_c = i * B_c; b_c < min((i + 1) * B_c, s); b_c++)
                {
                    int b_c_local = b_c - i * B_c;
                    float curr_l = l;
                    float curr_m = m;
                    float Sij = 0.0f;

                    for (int k_dim = 0; k_dim < d_head; k_dim++)
                    {
                        Sij += tau * (Q_shared[thread_i * d_head + k_dim] * K_shared[b_c_local * d_head + k_dim]);
                    }

                    float new_m = fmaxf(curr_m, Sij);
                    float exp_Sij = expf(Sij - new_m);
                    float exp_max = expf(curr_m - new_m);
                    float new_l = curr_l * exp_max + exp_Sij;

                    // Update output Oi += softmax * Vj
                    for (int v_dim = 0; v_dim < d_head; v_dim++)
                    {
                        out[block_b * h * s * d_head + block_h * s * d_head + r_idx * d_head + v_dim] =
                            out[block_b * h * s * d_head + block_h * s * d_head + r_idx * d_head + v_dim] * (curr_l * exp_max / new_l) + (exp_Sij / new_l) * V_shared[b_c_local * d_head + v_dim];
                    }

                    l = new_l;
                    m = new_m;
                }
            }
            __syncthreads();
        }

        if (r_idx < s)
        {
            l_hbm[block_b * h * s + block_h * s + r_idx] = l;
            m_hbm[block_b * h * s + block_h * s + r_idx] = m;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_attention_forwad(torch::Tensor q, torch::Tensor k, torch::Tensor v, float tau)
{
    int b = q.size(0);
    int h = q.size(1);
    int s = q.size(2);
    int d_head = q.size(3);

    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int maxSharedMemPerBlock;
    cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);

    int B_r = maxSharedMemPerBlock / (3 * d_head * sizeof(float));
    int B_c = maxSharedMemPerBlock / (3 * d_head * sizeof(float));

    int T_r = (s + B_r - 1) / B_r;
    int T_c = (s + B_c - 1) / B_c;

    torch::Tensor out = torch::zeros_like(q);

    dim3 grid(b, h);
    dim3 block(B_r);
    int shared_mem_size = (B_c * d_head * sizeof(float)) * 3; 

    auto l_hbm = torch::zeros({b, h, s}, torch::CUDA(torch::kFloat32));
    auto m_hbm = torch::zeros({b, h, s}, torch::CUDA(torch::kFloat32));

    flash_attn_forward_kernel<<<grid, block, shared_mem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        T_r,
        T_c,
        tau,
        l_hbm.data_ptr<float>(),
        m_hbm.data_ptr<float>(),
        b,
        h,
        s,
        d_head,
        B_r,
        B_c);

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(out, l_hbm, m_hbm);
}