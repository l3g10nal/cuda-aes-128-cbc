# AES128 CUDA Implementation

This repository contains a CUDA implementation of AES128 in CBC mode. It includes a stress test to measure the encryption and decryption performance and a test file to compare the CUDA implementation against OpenSSL's implementation.

Without any additional modifications/actions I got the following specs for a RTX2070:

- Encryption hashes per second: 1.0229e+12 hashes/s
- Decryption hashes per second: 1.0081e+12 hashes/s

PS! Keep in mind that any extra functions will slow down the process significantly.

PS2! Currently tested on Windows but should work the same way if not better on UNIX. In any case extreme stress tests have not be mayd so **use at your own risk, no warraties/guarantees**

ETH Donations welcome - 0x251a005aafd071aF6bD86F317840E5D8BBdAe402

## Files

- `aes128.h`: Header file containing function declarations.
- `aes128.cu`: Source file with AES128 implementation.
- `test_against_host.cu`: Compares CUDA AES128 implementation with OpenSSL's implementation.
- `stress_test_cuda.cu`: Stress test to measure encryption and decryption speeds.
- `compile_stress_test.bat`: Windows batch script to compile the stress test.
- `unix_build.sh`: Unix shell script to compile and link the files.

## Prerequisites

### Windows

1. **CUDA Toolkit**: Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
2. **OpenSSL**: Download and install OpenSSL for Windows. Make sure to set the environment variables `OPENSSL_INC` and `OPENSSL_LIB` to the OpenSSL include and library paths.
3. **Visual Studio**: Install [Visual Studio](https://visualstudio.microsoft.com/) with C++ development tools.

### Unix (Linux/MacOS)

1. **CUDA Toolkit**: Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
2. **OpenSSL**: Install OpenSSL. On Ubuntu, you can install it using:
		```sh
   sudo apt-get install libssl-dev```
3. **GCC**: Make sure GCC is installed on your system.

## Compilation Instructions

### Windows

#### Compile and Run Stress Test

1. Open a Command Prompt and navigate to the repository directory.
2. Run the following command to compile the stress test:
    ```
    compile_stress_test.bat
    ```
3. Execute the compiled stress test:
    ```
    stress_test_cuda.exe
    ```

#### Compile and Run Test Against Host

1. Open a Command Prompt and navigate to the repository directory.
2. Run the following command to compile the test against host(worked at least for me):
    ```
    nvcc -o test_against_host aes128.cu test_against_host.cu -I"C:\OpenSSL-Win64\include" -L"C:\OpenSSL-Win64\lib\VC\x64\MD" "C:\OpenSSL-Win64\lib\VC\x64\MD\libssl.lib" "C:\OpenSSL-Win64\lib\VC\x64\MD\libcrypto.lib"
    ```
3. Execute the compiled test:
    ```
    .\test_against_host.exe
    ```

### Unix

#### Compile and Run Stress Test

1. Open a terminal and navigate to the repository directory.
2. Run the following command to compile and link the files:
    ```
    ./unix_build.sh
    ```
3. Execute the compiled stress test:
    ```
    ./stress_test_cuda
    ```

## License

This project is licensed under the MIT License.
