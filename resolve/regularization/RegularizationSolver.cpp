/**
 * @file RegularizationSolver.cpp
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @brief Implementation of Regularization for LinSolverIterative classes.
 * 
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

#include <resolve/regularization/RegularizationSolver.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/MemoryUtils.hpp>

#include <Eigen/Dense>
#include <Eigen/src/SVD/JacobiSVD.h>

namespace ReSolve
{
  using out = io::Logger;

  RegularizationSolver::RegularizationSolver()
  {
  }

  RegularizationSolver::~RegularizationSolver()
  {
  }
  

  /**
    * @brief Performs Givens Rotations to perform regularization and solves for y
    *
    * This function carries out Givens Rotations to turn the stacked regularization problem
    * into an upper triangular problem and then backsolve for y for the following equation:
    *   x_n = x_0 + V_n @ y_n
    *
    * @param[in] H - The hessenberg matrix
    * @param[in] rs - The residual
    * @param[in] y - the regularized solution
    * @param[in] lambda - the regularization parameter
    * @param[in] n - The size of the hessenberg matrix
    * @param[in] restart - The number of iterations before gmres restart
  */
  int RegularizationSolver::regularize(real_type* H, real_type* rs, real_type* y, real_type reg_param, int n, index_type restart)
  { 
    real_type h_H_temp[n * n];
    real_type h_rs_temp[n];

    real_type reg_i = 0.0;
    real_type reg_j = std::sqrt(reg_param);

    real_type gamma = 0.0;
    real_type reg_c = 0.0;
    real_type reg_s = 0.0;

    int k = 0;
    int k1 = 0;
    real_type t = 0.0;

    // First we apply regularization to both the hessenberg matrix and the residuals
    // We save the regularized hessenberg matrix in h_H_temp
    for (int i = 0; i < n; i++) {
      reg_i = H[i * (restart + 1) + i];
      gamma = std::sqrt(reg_i * reg_i + reg_j * reg_j);

      // Rotation
      reg_c = reg_i/gamma;
      reg_s = reg_j/gamma;

      // Rotate all elements in the row and save in h_H_temp
      h_H_temp[i * n + i] = reg_c * reg_i + reg_s * reg_j;
      for (int j = i + 1; j < n; j++) {
        h_H_temp[j * n + i] = reg_c * H[j * (restart + 1) + i];
      }

      // Now rotate the residual vector and save in h_rs_temp
      h_rs_temp[i] = reg_c * rs[i];
    }

    // Now backsolve using h_H_temp and h_rs_temp to obtain y
    y[n-1] = h_rs_temp[n-1] / h_H_temp[n*n - 1];
    for(int ii = 2; ii <= n; ii++) {
      k = n - ii;
      k1 = k + 1;
      t = h_rs_temp[k];
      for (int j = k1; j < n; j++) {
        t -= h_H_temp[j * n + k] * y[j];
      }
      y[k] = t / h_H_temp[k * n + k];
    }
  
    return 0;
  }

  /**
    * @brief Generalized Cross Validation method to calculate the regularization parameter
    *
    * This function calculates the regularizaton parameter
    *
    * Requires use of the Eigen Library to calculate the SVD of H
    *
    * @param[in] H - The hessenberg matrix
    * @param[in] rs - The residual
    * @param[in] reg_param - the regularization parameter
    * @param[in] n - The size of the hessenberg matrix
    * @param[in] restart - Necessary to index into H properly
    * @param[in] tol - Tolerance for the golden section search
    * @param[in] max_iter - Maximum iterations for the golden section search
    */
  int RegularizationSolver::gcv(real_type* H, real_type* rs, real_type* reg_param, int n, index_type restart, real_type tol, real_type max_iter) 
  {
    // Create the Eigen Matrix version of the Hessenberg Matrix and b for calculations
    Eigen::Map<const Eigen::Matrix<real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eigen_H_(H, restart + 1, restart);
    Eigen::Map<const Eigen::Matrix<real_type, Eigen::Dynamic, 1>> eigen_b_(rs, restart + 1);

    // Calculate the SVD
    Eigen::JacobiSVD<Eigen::Matrix<real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> svd_H_(eigen_H_.topLeftCorner(n + 1, n), Eigen::ComputeFullU);
    Eigen::Matrix<real_type, Eigen::Dynamic, 1> H_s_(svd_H_.singularValues());
    Eigen::Matrix<real_type, Eigen::Dynamic, Eigen::Dynamic> H_U_(svd_H_.matrixU());

    // Square the singular values
    H_s_ = H_s_.array().square();

    // Obtain beta vector
    Eigen::Matrix<real_type, Eigen::Dynamic, 1> beta_(n);
    beta_.setZero();
    beta_ = (H_U_.transpose() * eigen_b_.head(n + 1)).eval();
    beta_ = beta_.head(H_s_.rows()).eval();

    // Golden Section Search to minimize gcv

    real_type phi_ = (std::sqrt(5) - 1)/2;
    real_type gcv_f1_ = 0.0;
    real_type gcv_f2_ = 0.0;
    gcv_a_ = 1e-09;
    gcv_b_ = 1e2;

    real_type x1_ = gcv_a_ + (1-phi_) * (gcv_b_ - gcv_a_);
    gcv_func(&gcv_f1_, x1_, H_s_.data(), beta_.data(), n);

    real_type x2_ = gcv_a_ + (phi_) * (gcv_b_ - gcv_a_);
    gcv_func(&gcv_f2_, x2_, H_s_.data(), beta_.data(), n);

    real_type n_iter = 0;

    while(std::abs(gcv_b_ - gcv_a_) > tol && n_iter < max_iter) {
      if (gcv_f1_ > gcv_f2_) {
        gcv_a_ = x1_;
        x1_ = x2_;
        gcv_f1_ = gcv_f2_;
        x2_ = gcv_a_ + phi_ * (gcv_b_ - gcv_a_);
        gcv_func(&gcv_f2_, x2_, H_s_.data(), beta_.data(), n);

      }

      else {
        gcv_b_ = x2_;
        x2_ = x1_;
        gcv_f2_ = gcv_f1_;
        x1_ = gcv_a_ + (1 - phi_) * (gcv_b_ - gcv_a_);
        gcv_func(&gcv_f1_, x1_, H_s_.data(), beta_.data(), n);
      }
        n_iter += 1;  
    }

    *reg_param = (gcv_b_ + gcv_a_)/2;
    return 0;
  }

  int RegularizationSolver::gcv_func(real_type* f, real_type reg_param, real_type* s2, real_type* beta, int n) 
  {
    Eigen::Map<const Eigen::Matrix<real_type, Eigen::Dynamic, 1>> s2_(s2, n);
    Eigen::Map<const Eigen::Matrix<real_type, Eigen::Dynamic, 1>> beta_(beta, n);
    
    Eigen::Matrix<real_type, Eigen::Dynamic, 1> filter_factors_(n);
    Eigen::Matrix<real_type, Eigen::Dynamic, 1> gcv_num_(n);
    filter_factors_ = (reg_param * reg_param)/(s2_.array() + reg_param);
    gcv_num_ = filter_factors_.array() * beta_.array();
    *f = (gcv_num_.norm() * gcv_num_.norm())/(filter_factors_.sum() * filter_factors_.sum());
    return 0;
  }

} // namespace ReSolve