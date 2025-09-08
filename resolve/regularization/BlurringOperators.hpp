/**
 * @file BlurringOperators.hpp
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @brief Declaration of BlurringOperators class
 * 
 */

#pragma once
#include <iostream>
#include <cassert>

#include <resolve/Common.hpp>

namespace ReSolve 
{
  /**
   * @brief Blurring Operators
   * 
   * @author Jeffery Zhang (jefferyz@vt.edu)
   * 
   */
  class BlurringOperators
  {
    public:
      BlurringOperators();
      ~BlurringOperators();
      
    /**
    * @brief Creates the blurring operator
    *
    * This function creates a column major blurring operator
    *
    * @param[in] A - The blurring matrix
    * @param[in] m - the number of rows of A
    * @param[in] n - the number of columns of A
    * @param[in] spread_a - the vertical spread
    * @param[in] spread_b - the horizontal spread
    */
      int build_A(real_type* A, int m, int n, real_type spread_a, real_type spread_b);

  };
} 
