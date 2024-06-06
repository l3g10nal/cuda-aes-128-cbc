@echo off
rem Set the OpenSSL include and library paths
set OPENSSL_INC="C:\OpenSSL-Win64\include"
set OPENSSL_LIB="C:\OpenSSL-Win64\lib\VC\x64\MD"

rem Step 1: Compile aes128.cu to generate a device object file
echo Compiling aes128.cu...
nvcc -dc -o aes128.o aes128.cu -I%OPENSSL_INC%
if %errorlevel% neq 0 (
    echo Compilation of aes128.cu failed.
    exit /b %errorlevel%
)

rem Step 2: Compile test_against_host.cu to generate another device object file
echo Compiling test_against_host.cu...
nvcc -dc -o test_against_host.o test_against_host.cu -I%OPENSSL_INC%
if %errorlevel% neq 0 (
    echo Compilation of test_against_host.cu failed.
    exit /b %errorlevel%
)

rem Step 3: Link the object files together to create the final executable
echo Linking object files...
nvcc -o test_against_host aes128.o test_against_host.o -L%OPENSSL_LIB% -lssl -lcrypto
if %errorlevel% neq 0 (
    echo Linking failed.
    exit /b %errorlevel%
)

echo Compilation and linking completed successfully.
