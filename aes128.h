#ifndef AES128_H
#define AES128_H

#include <cuda_runtime.h>
#include <string>

__global__ void run_cbc_encrypt(const unsigned char* plaintext, unsigned char* ciphertext, const unsigned char* key, const unsigned char* iv);
__global__ void run_cbc_decrypt(const unsigned char* ciphertext, unsigned char* plaintext, const unsigned char* key, const unsigned char* iv);

// Declare the AES functions void AES128_key_expansion(const unsigned char* key, unsigned char* round_keys);
__device__ void AES128_CBC_encrypt(const unsigned char* plaintext, unsigned char* ciphertext, const unsigned char* key, const unsigned char* iv, int length);
__device__ void AES128_CBC_decrypt(const unsigned char* ciphertext, unsigned char* plaintext, const unsigned char* key, const unsigned char* iv, int length);

void print_array(const std::string& label, const unsigned char* array, int length);

#endif // AES128_H
