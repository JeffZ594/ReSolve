/**
 * @file BlurringOperators.cpp
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @brief Implementation of BlurringOperators class
 * 
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

#include <resolve/regularization/BlurringOperators.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/Common.hpp>

namespace ReSolve 
{
  using out = io::Logger;

  BlurringOperators::BlurringOperators()
  {
  }

  BlurringOperators::~BlurringOperators()
  {
  }
      
  /**
  * @brief Creates the blurring operator
  *
  * This function creates a column major matrix of a row major blurring operator with a gaussian spread
  *
  * @param[in] A - The blurring matrix
  * @param[in] m - the number of rows of A
  * @param[in] n - the number of columns of A
  * @param[in] spread_a - the vertical spread
  * @param[in] spread_b - the horizontal spread
  */
  int BlurringOperators::build_A(real_type* A, int m, int n, real_type spread_a, real_type spread_b)
  {
    int size = m * n;
    int count = 0;

    // loop over every pixel of A 
    for (int a_i = 0; a_i < m; a_i++) {
      for (int a_j = 0; a_j < n; a_j++) {
        // fill column of A
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            double di = (i - a_i) / spread_a;
            double dj = (j - a_j) / spread_b;
            double v = std::exp(-(di*di + dj*dj) / 2.0);

            if (v < 0.0001) {
              v = 0.0;
            } 

            int row = i * n + j; // flatten (i,j) into row index
            A[row * size + count] = v;  // column-major layout
          }
        }
        count++;
      }
    }
    // normalize with the center column
    int mid_col = (m/2) * n + (n/2);
    double sum = 0.0;
    for (int row = 0; row < size; row++) {
      sum += A[row * size + mid_col];
    }
    for (int k = 0; k < size * size; k++) {
      A[k] /= sum;
    }
    return 0;
  }
      
} 