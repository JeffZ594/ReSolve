#!/bin/bash

# Make the test in build folder
cd ../build/tests/functionality
make rand_gmres_test.exe

# Run the gmres test
./rand_gmres_test.exe