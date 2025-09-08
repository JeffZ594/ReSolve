#!/bin/bash
cd ../ && make test_hybrid_gmres_export.exe
cd trips_code
../test_hybrid_gmres_export.exe input_files/trips_1d_deblur_A.mtx input_files/trips_1d_deblur_b_vec.mtx input_files/trips_1d_deblur_A.mtx input_files/trips_1d_deblur_x_true.mtx
mv Hybrid_ABGMRES_history.csv csv_files/