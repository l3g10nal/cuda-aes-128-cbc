#include "aes128.h"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 768

// Kernel to perform AES CBC encryption
__global__ void run_stress_test_encrypt(const unsigned char* plaintext, unsigned char* ciphertext, const unsigned char* key, const unsigned char* iv, size_t iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char local_plaintext[16];
    unsigned char local_ciphertext[16];
    unsigned char local_key[16];
    unsigned char local_iv[16];

    for (int i = 0; i < 16; ++i) {
        local_plaintext[i] = plaintext[i];
        local_key[i] = key[i];
        local_iv[i] = iv[i];
    }

    for (size_t i = 0; i < iterations; ++i) {
        AES128_CBC_encrypt(local_plaintext, local_ciphertext, local_key, local_iv, 16);
    }

    for (int i = 0; i < 16; ++i) {
        ciphertext[idx * 16 + i] = local_ciphertext[i];
    }
}

// Kernel to perform AES CBC decryption
__global__ void run_stress_test_decrypt(const unsigned char* ciphertext, unsigned char* plaintext, const unsigned char* key, const unsigned char* iv, size_t iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char local_ciphertext[16];
    unsigned char local_plaintext[16];
    unsigned char local_key[16];
    unsigned char local_iv[16];

    for (int i = 0; i < 16; ++i) {
        local_ciphertext[i] = ciphertext[i];
        local_key[i] = key[i];
        local_iv[i] = iv[i];
    }

    for (size_t i = 0; i < iterations; ++i) {
        AES128_CBC_decrypt(local_ciphertext, local_plaintext, local_key, local_iv, 16);
    }

    for (int i = 0; i < 16; ++i) {
        plaintext[idx * 16 + i] = local_plaintext[i];
    }
}

int main(int argc, char* argv[]) {
    unsigned char key[16];
    unsigned char iv[16];
    unsigned char plaintext[16];
    unsigned char* d_key, * d_iv, * d_plaintext, * d_ciphertext, * d_decrypted;

    std::string key_str = "86c170c59e856464dba8d55234e8a941";
    std::string iv_str = "edfc22e6ac7e1793171a888451afdf30";
    std::string pt_str = "00112233445566778899aabbccddeeff";

    for (int i = 0; i < 16; ++i) {
        key[i] = std::stoi(key_str.substr(2 * i, 2), nullptr, 16);
        iv[i] = std::stoi(iv_str.substr(2 * i, 2), nullptr, 16);
        plaintext[i] = std::stoi(pt_str.substr(2 * i, 2), nullptr, 16);
    }

    cudaMalloc((void**)&d_key, 16 * sizeof(unsigned char));
    cudaMalloc((void**)&d_iv, 16 * sizeof(unsigned char));
    cudaMalloc((void**)&d_plaintext, 16 * sizeof(unsigned char));
    cudaMalloc((void**)&d_ciphertext, THREADS_PER_BLOCK * 16 * sizeof(unsigned char));
    cudaMalloc((void**)&d_decrypted, THREADS_PER_BLOCK * 16 * sizeof(unsigned char));

    cudaMemcpy(d_key, key, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, iv, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_plaintext, plaintext, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    size_t iterations = 0;
    size_t num_blocks = 1; // Adjust this based on your GPU's capability
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;

    // Encryption stress test for 15 seconds
    do {
        run_stress_test_encrypt<<<num_blocks, THREADS_PER_BLOCK>>>(d_plaintext, d_ciphertext, d_key, d_iv, 1000);
        cudaDeviceSynchronize();
        iterations += num_blocks * THREADS_PER_BLOCK * 1000;
        end = std::chrono::high_resolution_clock::now();
    } while (std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < 15);

    double encryption_hashes_per_second = static_cast<double>(iterations) / 15.0;

    std::cout << "Encryption hashes per second: " << encryption_hashes_per_second << " hashes/s" << std::endl;

    iterations = 0;
    start = std::chrono::high_resolution_clock::now();

    // Decryption stress test for 15 seconds
    do {
        run_stress_test_decrypt<<<num_blocks, THREADS_PER_BLOCK>>>(d_ciphertext, d_decrypted, d_key, d_iv, 1000);
        cudaDeviceSynchronize();
        iterations += num_blocks * THREADS_PER_BLOCK * 1000;
        end = std::chrono::high_resolution_clock::now();
    } while (std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < 15);

    double decryption_hashes_per_second = static_cast<double>(iterations) / 15.0;

    std::cout << "Decryption hashes per second: " << decryption_hashes_per_second << " hashes/s" << std::endl;

    cudaFree(d_key);
    cudaFree(d_iv);
    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);
    cudaFree(d_decrypted);

    return 0;
}
