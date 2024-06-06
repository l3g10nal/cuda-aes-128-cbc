#!/bin/bash

# Set the OpenSSL include and library paths
OPENSSL_INC="/usr/include/openssl"
OPENSSL_LIB="/usr/lib"

# Step 1: Compile aes128.cu to generate a device object file
echo "Compiling aes128.cu..."
nvcc -dc -o aes128.o aes128.cu -I${OPENSSL_INC}
if [ $? -ne 0 ]; then
    echo "Compilation of aes128.cu failed."
    exit 1
fi

# Step 2: Compile stress_test_cuda.cu to generate another device object file
echo "Compiling stress_test_cuda.cu..."
nvcc -dc -o stress_test_cuda.o stress_test_cuda.cu -I${OPENSSL_INC}
if [ $? -ne 0 ]; then
    echo "Compilation of stress_test_cuda.cu failed."
    exit 1
fi

# Step 3: Link the object files together to create the final executable
echo "Linking object files..."
nvcc -o stress_test_cuda aes128.o stress_test_cuda.o -L${OPENSSL_LIB} -lssl -lcrypto
if [ $? -ne 0 ]; then
    echo "Linking failed."
    exit 1
fi

echo "Compilation and linking completed successfully."
