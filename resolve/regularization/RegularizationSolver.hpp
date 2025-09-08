/**
 * @file RegularizationSolver.hpp
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @brief Declaration of RegularizationSolver class
 * 
 */

#pragma once
#include <iostream>
#include <cassert>

#include <resolve/Common.hpp>

namespace ReSolve 
{
  /**
   * @brief Regularization solver
   * 
   * @author Jeffery Zhang (jefferyz@vt.edu)
   * 
   * @note Assumes Simple Identity operator for stacked matrix
   *       Assumes upper triangular Hessenberg Matrix and rotated residuals
   */
  class RegularizationSolver
  {
    public:
      RegularizationSolver();
      ~RegularizationSolver();
      
    /**
    * @brief Performs Givens Rotations to perform regularization
    *
    * This function carries out Givens Rotations to turn the stacked regularization problem
    * into an upper triangular problem
    *
    * @param[in] H - The hessenberg matrix
    * @param[in] rs - The residual
    * @param[in] y - the regularized solution
    * @param[in] method - the method to calculate the regularization parameter
    * @param[in] n - The size of the hessenberg matrix
    * @param[in] restart - Necessary to index into H properly
    */
      int regularize(real_type* H, real_type* rs, real_type* y, real_type reg_param, int n, index_type restart);
    
    /**
    * @brief Generalized Cross Validation method to calculate the regularization parameter
    *
    * This function calculates the regularizaton parameter
    * 
    * Called when method == 0
    *
    * Requires use of the Eigen Library to calculate the SVD of H
    *
    * @param[in] H - The hessenberg matrix
    * @param[in] rs - The residual
    * @param[in] reg_param - the regularization parameter
    * @param[in] n - The size of the hessenberg matrix
    * @param[in] restart - Necessary to index into H properly
    */
      int gcv(real_type* H, real_type* rs, real_type* reg_param, int n, index_type restart, real_type tol, real_type max_iter);

    private:
    /**
    * @brief Helper function to calculate the gcv at a given reg_param
    *
    * Passing s2 and beta is convoluted as #define <Eigen/Dense> throws an error in the hpp file only
    *
    */
      int gcv_func(real_type* f, real_type reg_param, real_type* s2, real_type* beta, int n);

      real_type gcv_a_{1e-09};
      real_type gcv_b_{1e2};

  };
} 
