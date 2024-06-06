#include "aes128.h"
#include <iostream>
#include <openssl/aes.h>

void host_aes_encrypt(const unsigned char* key, const unsigned char* iv, const unsigned char* plaintext, unsigned char* ciphertext) {
    AES_KEY encrypt_key;
    AES_set_encrypt_key(key, 128, &encrypt_key);
    AES_cbc_encrypt(plaintext, ciphertext, 16, &encrypt_key, const_cast<unsigned char*>(iv), AES_ENCRYPT);
}

void host_aes_decrypt(const unsigned char* key, const unsigned char* iv, const unsigned char* ciphertext, unsigned char* plaintext) {
    AES_KEY decrypt_key;
    AES_set_decrypt_key(key, 128, &decrypt_key);
    AES_cbc_encrypt(ciphertext, plaintext, 16, &decrypt_key, const_cast<unsigned char*>(iv), AES_DECRYPT);
}

int main(int argc, char* argv[]) {
    unsigned char key[16];
    unsigned char iv[16];
    unsigned char iv_decrypt[16];  // Separate IV for decryption
    unsigned char plaintext[16];
    unsigned char ciphertext_gpu[16];
    unsigned char ciphertext_host[16];
    unsigned char decrypted_gpu[16];
    unsigned char decrypted_host[16];
    unsigned char* d_key, * d_iv, * d_plaintext, * d_ciphertext, * d_decrypted;

    std::string key_str = "86c170c59e856464dba8d55234e8a941";
    std::string iv_str = "edfc22e6ac7e1793171a888451afdf30";
    std::string pt_str = "00112233445566778899aabbccddeeff";

    for (int i = 0; i < 16; ++i) {
        key[i] = std::stoi(key_str.substr(2 * i, 2), nullptr, 16);
        iv[i] = std::stoi(iv_str.substr(2 * i, 2), nullptr, 16);
        iv_decrypt[i] = iv[i];  // Initialize iv_decrypt with the same IV
        plaintext[i] = std::stoi(pt_str.substr(2 * i, 2), nullptr, 16);
    }

    cudaMalloc((void**)&d_key, 16 * sizeof(unsigned char));
    cudaMalloc((void**)&d_iv, 16 * sizeof(unsigned char));
    cudaMalloc((void**)&d_plaintext, 16 * sizeof(unsigned char));
    cudaMalloc((void**)&d_ciphertext, 16 * sizeof(unsigned char));
    cudaMalloc((void**)&d_decrypted, 16 * sizeof(unsigned char));

    cudaMemcpy(d_key, key, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, iv, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_plaintext, plaintext, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Encryption
    run_cbc_encrypt<<<1, 1>>>(d_plaintext, d_ciphertext, d_key, d_iv);
    cudaMemcpy(ciphertext_gpu, d_ciphertext, 16 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    print_array("Encrypted block (GPU)", ciphertext_gpu, 16);

    host_aes_encrypt(key, iv, plaintext, ciphertext_host);
    print_array("Encrypted block (Host)", ciphertext_host, 16);

    // Decryption
    cudaMemcpy(d_ciphertext, ciphertext_gpu, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, iv, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);  // Reset IV for decryption
    run_cbc_decrypt<<<1, 1>>>(d_ciphertext, d_decrypted, d_key, d_iv);
    cudaMemcpy(decrypted_gpu, d_decrypted, 16 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    print_array("Decrypted block (GPU)", decrypted_gpu, 16);

    // Reset IV for host decryption
    for (int i = 0; i < 16; ++i) {
        iv_decrypt[i] = iv[i];
    }
    host_aes_decrypt(key, iv_decrypt, ciphertext_host, decrypted_host);
    print_array("Decrypted block (Host)", decrypted_host, 16);

    cudaFree(d_key);
    cudaFree(d_iv);
    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);
    cudaFree(d_decrypted);

    return 0;
}
