/**
 * @file LinSolverIterativeHybridABGMRES.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @brief Implementation of LinSolverIterativeHybridABGMRES class
 *
 */
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <fstream>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include "LinSolverIterativeHybridABGMRES.hpp"

namespace ReSolve
{
  using out = io::Logger;

  LinSolverIterativeHybridABGMRES::LinSolverIterativeHybridABGMRES(MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs)
  {
    matrix_handler_ = matrix_handler;
    vector_handler_ = vector_handler;
    GS_ = gs;
    setMemorySpace();
    initParamList();
  }

  // Constructor with regularization
  LinSolverIterativeHybridABGMRES::LinSolverIterativeHybridABGMRES(MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs,
                                                     RegularizationSolver* rs)
  {
    matrix_handler_ = matrix_handler;
    vector_handler_ = vector_handler;
    GS_ = gs;
    RS_ = rs;
    setMemorySpace();
    initParamList();
  }

  LinSolverIterativeHybridABGMRES::LinSolverIterativeHybridABGMRES(index_type     restart,
                                                     real_type      tol,
                                                     index_type     maxit,
                                                     index_type     conv_cond,
                                                     MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs)
  {
    // Base class settings here (to be removed when solver parameter settings are implemented)
    tol_ = tol;
    maxit_= maxit;
    restart_ = restart;
    conv_cond_ = conv_cond;

    matrix_handler_ = matrix_handler;
    vector_handler_ = vector_handler;
    GS_ = gs;
    setMemorySpace();
    initParamList();
  }

  LinSolverIterativeHybridABGMRES::~LinSolverIterativeHybridABGMRES()
  {
    if (is_solver_set_) {
      freeSolverData();
    }
  }

  /**
   * @brief Set pointer to system matrix and preconditioner matrix and 
   *        allocate solver data.
   *
   * @param[in] A - Sparse system matrix, 
   * @param[in] B - Preconditioner matrix
   *
   * @pre A and B are valid sparse matrices
   *
   * @post A_ == A
   * @post B_ == B
   * @post Solver data allocated.
   */
  int LinSolverIterativeHybridABGMRES::setup(matrix::Sparse* A, matrix::Sparse* B)
  {
    // If A_ is already set, then report error and exit.
    if (n_ != A->getNumRows()) {
      if (is_solver_set_) {
        out::warning() << "Matrix size changed. Reallocating solver ...\n";
        freeSolverData();
        is_solver_set_ = false;
      }
    }

    // Set pointer to matrix A and the matrix size.
    A_ = A;
    n_ = A->getNumRows();

    // Set pointer to matrix B
    B_ = B;

    // Allocate solver data.
    if (!is_solver_set_) {
      allocateSolverData();
      is_solver_set_ = true;
    }

    GS_->setup(n_, restart_);

    return 0;
  }

  /**
  * @brief Hybrid ABGMRES where a regularization parameter is calculated at each iteration
  * 
  * Currently has no stopping rule so will run for max iterations
  */
  int  LinSolverIterativeHybridABGMRES::solve(vector_type* rhs, vector_type* x)
  {
    using namespace constants;

    //io::Logger::setVerbosity(io::Logger::EVERYTHING);

    int outer_flag = 1;
    int inner_flag = 1;
    int stop_flag = 1;
    int i  = 0;
    int it = 0;
    int j  = 0;
    int k1 = 0;

    real_type t = 0.0;
    real_type rnorm = 0.0;
    real_type bnorm = 0.0;

    real_type* y = new real_type[restart_ + 1]();

    real_type reg_param = 0.0;

    // Used to track the size of the Hessenberg matrix
    int H_n_ = 0;
    
    vector_type* vec_prev_x_ = new vector_type(n_);

    vector_type* vec_v = new vector_type(n_);
    vector_type* vec_z = new vector_type(n_);
    vector_type* vec_norm = new vector_type(n_);
    vector_type* vec_x_i = new vector_type(n_);

    vec_x_i->setToZero(memspace_);
    vec_prev_x_->setToZero(memspace_);
    //V[0] = b-A*x_0
    //debug
    vec_Z_->setToZero(memspace_);
    vec_V_->setToZero(memspace_);

    rhs->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);
    matrix_handler_->matvec(A_, x, vec_V_, &MINUS_ONE, &ONE, memspace_);
    rnorm = 0.0;
    bnorm = vector_handler_->dot(rhs, rhs, memspace_);
    rnorm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
    //rnorm = ||V_1||
    rnorm = std::sqrt(rnorm);
    bnorm = std::sqrt(bnorm);
    io::Logger::misc() << "it 0: norm of residual "
                       << std::scientific << std::setprecision(16)
                       << rnorm << " Norm of rhs: " << bnorm << "\n";
    initial_residual_norm_ = rnorm;

    while(outer_flag) {

      // normalize first vector
      t = 1.0 / rnorm;
      vector_handler_->scal(&t, vec_V_, memspace_);
      // initialize norm history
      h_rs_[0] = rnorm;
      i = -1;
      inner_flag = 1;

      while((inner_flag) && (i + 1 < restart_)) {
        i++;
        it++;

        // Z_i = B * V_i
        vec_v->setData( vec_V_->getData(i, memspace_), memspace_);
  
        vec_z->setData( vec_Z_->getData(0, memspace_), memspace_);
        matrix_handler_->matvec(B_, vec_v, vec_z, &ONE, &ZERO, memspace_);
        
        mem_.deviceSynchronize();

        // V_{i+1}=A*Z_i

        vec_v->setData( vec_V_->getData(i + 1, memspace_), memspace_);

        matrix_handler_->matvec(A_, vec_z, vec_v, &ONE, &ZERO, memspace_);

        // orthogonalize V[i+1], form a column of h_H_

        GS_->orthogonalize(n_, vec_V_, h_H_, i);

        // Increment H_n_
        H_n_++;
        if (H_n_ > restart_) {
          H_n_ = 1;
        }

        // Givens Rotations on Previous Rows
        if (i != 0) {
          for (index_type k = 1; k <= i; k++) {
            k1 = k - 1;
            t = h_H_[i * (restart_ + 1) + k1];
            h_H_[i * (restart_ + 1) + k1] = h_c_[k1] * t + h_s_[k1] * h_H_[i * (restart_ + 1) + k];
            h_H_[i * (restart_ + 1) + k] = -h_s_[k1] * t + h_c_[k1] * h_H_[i * (restart_ + 1) + k];
          }
        } // if i!=0
        real_type Hii = h_H_[i * (restart_ + 1) + i];
        real_type Hii1 = h_H_[(i) * (restart_ + 1) + i + 1];
        real_type gam = std::sqrt(Hii * Hii + Hii1 * Hii1);

        if(std::abs(gam - ZERO) <= MACHINE_EPSILON) {
          gam = MACHINE_EPSILON;
        }

        /* next Given's rotation */
        h_c_[i] = Hii / gam;
        h_s_[i] = Hii1 / gam;
        h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
        h_rs_[i] = h_c_[i] * h_rs_[i];

        h_H_[(i) * (restart_ + 1) + (i)]     = h_c_[i] * Hii  + h_s_[i] * Hii1;
        h_H_[(i) * (restart_ + 1) + (i + 1)] = h_c_[i] * Hii1 - h_s_[i] * Hii;

         // Obtain parameter
        RS_ ->gcv(h_H_, h_rs_, &reg_param, H_n_, restart_, 1e-12, 100);

        // Call regularization solver
        RS_-> regularize(h_H_, h_rs_, y, reg_param, H_n_, restart_);

        // get solution
        vec_Z_->setToZero(memspace_);
        vec_z->setData( vec_Z_->getData(0, memspace_), memspace_);
        for (j = 0; j <= i; j++) {
          vec_v->setData( vec_V_->getData(j, memspace_), memspace_);
          vector_handler_->axpy(&y[j], vec_v, vec_z, memspace_);
        }

        // now multiply d_Z by B
        matrix_handler_->matvec(B_, vec_z, vec_x_i, &ONE, &ZERO, memspace_);

        // and add to x
        vector_handler_->axpy(&ONE, x, vec_x_i, memspace_);

        vec_norm->copyDataFrom(rhs, memspace_, memspace_);
        matrix_handler_->matvec(A_, vec_x_i, vec_norm, &MINUS_ONE, &ONE, memspace_);
        rnorm = vector_handler_->dot(vec_norm, vec_norm, memspace_);
        // rnorm = ||V_1||
        rnorm = std::sqrt(rnorm);

        // residual norm estimate
        io::Logger::misc() << "it: " << it << " --> norm of the residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";
        // check convergence
        if (i + 1 >= restart_ || !stop_flag || it >= maxit_) {
          inner_flag = 0;
        }
      } // inner while

      io::Logger::misc() << "End of cycle, ESTIMATED norm of residual "
                         << std::scientific << std::setprecision(16)
                         << rnorm << "\n";

      /* test solution */

      if(!stop_flag || it >= maxit_) {
        outer_flag = 0;
        x->copyDataFrom(vec_x_i, memspace_, memspace_);
      }

      if(!outer_flag) {
        final_residual_norm_ = rnorm;
        total_iters_ = it;
        io::Logger::misc() << "End of cycle, COMPUTED norm of residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";
      }
      if(outer_flag){
        rhs->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);
        matrix_handler_->matvec(A_, vec_x_i, vec_V_, &MINUS_ONE, &ONE, memspace_);
        rnorm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
        // rnorm = ||V_1||
        rnorm = std::sqrt(rnorm);
      }
    } // outer while
    return 0;
  }

  /**
  @brief Hybrid GMRES using discrepancy princple as a stopping rule

  Requires delta to be provided
  */
  int  LinSolverIterativeHybridABGMRES::solve(vector_type* rhs, vector_type* x, real_type delta)
  {
    using namespace constants;

    //io::Logger::setVerbosity(io::Logger::EVERYTHING);

    int outer_flag = 1;
    int inner_flag = 1;
    int stop_flag = 1;
    int i  = 0;
    int it = 0;
    int j  = 0;
    int k1 = 0;

    real_type t = 0.0;
    real_type rnorm = 0.0;
    real_type bnorm = 0.0;

    real_type* y = new real_type[restart_ + 1]();

    real_type reg_param = 0.0;

    // Used to track the size of the Hessenberg matrix
    int H_n_ = 0;
    
    vector_type* vec_prev_x_ = new vector_type(n_);

    vector_type* vec_v = new vector_type(n_);
    vector_type* vec_z = new vector_type(n_);
    vector_type* vec_norm = new vector_type(n_);
    vector_type* vec_x_i = new vector_type(n_);

    vec_x_i->setToZero(memspace_);
    vec_prev_x_->setToZero(memspace_);
    //V[0] = b-A*x_0
    //debug
    vec_Z_->setToZero(memspace_);
    vec_V_->setToZero(memspace_);

    rhs->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);
    matrix_handler_->matvec(A_, x, vec_V_, &MINUS_ONE, &ONE, memspace_);
    rnorm = 0.0;
    bnorm = vector_handler_->dot(rhs, rhs, memspace_);
    rnorm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
    //rnorm = ||V_1||
    rnorm = std::sqrt(rnorm);
    bnorm = std::sqrt(bnorm);
    io::Logger::misc() << "it 0: norm of residual "
                       << std::scientific << std::setprecision(16)
                       << rnorm << " Norm of rhs: " << bnorm << "\n";
    initial_residual_norm_ = rnorm;

    while(outer_flag) {

      // normalize first vector
      t = 1.0 / rnorm;
      vector_handler_->scal(&t, vec_V_, memspace_);
      // initialize norm history
      h_rs_[0] = rnorm;
      i = -1;
      inner_flag = 1;

      while((inner_flag) && (i + 1 < restart_)) {
        i++;
        it++;

        // Z_i = B * V_i
        vec_v->setData( vec_V_->getData(i, memspace_), memspace_);
  
        vec_z->setData( vec_Z_->getData(0, memspace_), memspace_);
        matrix_handler_->matvec(B_, vec_v, vec_z, &ONE, &ZERO, memspace_);
        
        mem_.deviceSynchronize();

        // V_{i+1}=A*Z_i

        vec_v->setData( vec_V_->getData(i + 1, memspace_), memspace_);

        matrix_handler_->matvec(A_, vec_z, vec_v, &ONE, &ZERO, memspace_);

        // orthogonalize V[i+1], form a column of h_H_

        GS_->orthogonalize(n_, vec_V_, h_H_, i);

        // Increment H_n_
        H_n_++;
        if (H_n_ > restart_) {
          H_n_ = 1;
        }

        // Givens Rotations on Previous Rows
        if (i != 0) {
          for (index_type k = 1; k <= i; k++) {
            k1 = k - 1;
            t = h_H_[i * (restart_ + 1) + k1];
            h_H_[i * (restart_ + 1) + k1] = h_c_[k1] * t + h_s_[k1] * h_H_[i * (restart_ + 1) + k];
            h_H_[i * (restart_ + 1) + k] = -h_s_[k1] * t + h_c_[k1] * h_H_[i * (restart_ + 1) + k];
          }
        } // if i!=0
        real_type Hii = h_H_[i * (restart_ + 1) + i];
        real_type Hii1 = h_H_[(i) * (restart_ + 1) + i + 1];
        real_type gam = std::sqrt(Hii * Hii + Hii1 * Hii1);

        if(std::abs(gam - ZERO) <= MACHINE_EPSILON) {
          gam = MACHINE_EPSILON;
        }

        /* next Given's rotation */
        h_c_[i] = Hii / gam;
        h_s_[i] = Hii1 / gam;
        h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
        h_rs_[i] = h_c_[i] * h_rs_[i];

        h_H_[(i) * (restart_ + 1) + (i)]     = h_c_[i] * Hii  + h_s_[i] * Hii1;
        h_H_[(i) * (restart_ + 1) + (i + 1)] = h_c_[i] * Hii1 - h_s_[i] * Hii;

         // Obtain parameter
        RS_ ->gcv(h_H_, h_rs_, &reg_param, H_n_, restart_, 1e-12, 100);

        // Call regularization solver
        RS_-> regularize(h_H_, h_rs_, y, reg_param, H_n_, restart_);

        // get solution
        vec_Z_->setToZero(memspace_);
        vec_z->setData( vec_Z_->getData(0, memspace_), memspace_);
        for (j = 0; j <= i; j++) {
          vec_v->setData( vec_V_->getData(j, memspace_), memspace_);
          vector_handler_->axpy(&y[j], vec_v, vec_z, memspace_);
        }

        // now multiply d_Z by B
        matrix_handler_->matvec(B_, vec_z, vec_x_i, &ONE, &ZERO, memspace_);

        // and add to x
        vector_handler_->axpy(&ONE, x, vec_x_i, memspace_);

        vec_norm->copyDataFrom(rhs, memspace_, memspace_);
        matrix_handler_->matvec(A_, vec_x_i, vec_norm, &MINUS_ONE, &ONE, memspace_);
        rnorm = vector_handler_->dot(vec_norm, vec_norm, memspace_);
        // rnorm = ||V_1||
        rnorm = std::sqrt(rnorm);

        if (rnorm < delta * 1.01) 
        {
          stop_flag = 0;
        }

        // residual norm estimate
        io::Logger::misc() << "it: " << it << " --> norm of the residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";
        // check convergence
        if (i + 1 >= restart_ || !stop_flag || it >= maxit_) {
          inner_flag = 0;
        }
      } // inner while

      io::Logger::misc() << "End of cycle, ESTIMATED norm of residual "
                         << std::scientific << std::setprecision(16)
                         << rnorm << "\n";

      /* test solution */

      if(!stop_flag || it >= maxit_) {
        outer_flag = 0;
        x->copyDataFrom(vec_x_i, memspace_, memspace_);
      }

      if(!outer_flag) {
        final_residual_norm_ = rnorm;
        total_iters_ = it;
        io::Logger::misc() << "End of cycle, COMPUTED norm of residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";
      }
      if(outer_flag){
        rhs->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);
        matrix_handler_->matvec(A_, vec_x_i, vec_V_, &MINUS_ONE, &ONE, memspace_);
        rnorm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
        // rnorm = ||V_1||
        rnorm = std::sqrt(rnorm);
      }
    } // outer while
    return 0;
  }


  /**
  * @brief A function used to save the residual and regularization parameter at each iteration to a csv
  *
  * Always goes to the end of restart
  */
  int  LinSolverIterativeHybridABGMRES::solveExport(vector_type* rhs, vector_type* x, vector_type* x_true)
  {
    using namespace constants;

    //io::Logger::setVerbosity(io::Logger::EVERYTHING);

    int outer_flag = 1;
    int inner_flag = 1;
    int stop_flag = 1;
    int i  = 0;
    int it = 0;
    int j  = 0;
    int k1 = 0;

    real_type t = 0.0;
    real_type rnorm = 0.0;
    real_type bnorm = 0.0;

    real_type* y = new real_type[restart_ + 1]();

    real_type reg_param = 0.0;

    // Used to track the size of the Hessenberg matrix
    int H_n_ = 0;
    
    real_type rel_error_norm_ = 0.0;
    real_type x_true_norm_ = 0.0;

    vector_type* vec_prev_x_ = new vector_type(n_);

    vector_type* vec_v = new vector_type(n_);
    vector_type* vec_z = new vector_type(n_);
    vector_type* vec_norm = new vector_type(n_);
    vector_type* vec_x_i = new vector_type(n_);
    vector_type* vec_rel_error = new vector_type(n_);

    std::ofstream file("Hybrid_ABGMRES_history.csv");
    if (!file.is_open()) {
      std::cerr << "Failed to open file.\n";
      return 1;
    }    

    file << "Iter,rel_residual,rel_error,reg_param" << "\n";
    file << std::scientific << std::setprecision(16);  

    vec_x_i->setToZero(memspace_);
    vec_prev_x_->setToZero(memspace_);

    //V[0] = b-A*x_0
    //debug
    vec_Z_->setToZero(memspace_);
    vec_V_->setToZero(memspace_);

    rhs->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);
    matrix_handler_->matvec(A_, x, vec_V_, &MINUS_ONE, &ONE, memspace_);
    rnorm = 0.0;
    bnorm = vector_handler_->dot(rhs, rhs, memspace_);
    rnorm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
    //rnorm = ||V_1||
    rnorm = std::sqrt(rnorm);
    bnorm = std::sqrt(bnorm);
    io::Logger::misc() << "it 0: norm of residual "
                       << std::scientific << std::setprecision(16)
                       << rnorm << " Norm of rhs: " << bnorm << "\n";
    initial_residual_norm_ = rnorm;

    // Calculate relative error
    vec_rel_error->setToZero(memspace_);
    vec_rel_error->copyDataFrom(x_true, memspace_, memspace_);
    vector_handler_->axpy(&MINUS_ONE, x, vec_rel_error, memspace_);

    rel_error_norm_ = vector_handler_->dot(vec_rel_error, vec_rel_error, memspace_);
    rel_error_norm_ = std::sqrt(rnorm);

    x_true_norm_ = vector_handler_->dot(x_true, x_true, memspace_);
    x_true_norm_ = std::sqrt(x_true_norm_);

    rel_error_norm_ = rel_error_norm_/x_true_norm_;

    // Write to file
    file << 0 << "," << rnorm << "," << rel_error_norm_ << "," << 0 << "\n";

    while(outer_flag) {

      // normalize first vector
      t = 1.0 / rnorm;
      vector_handler_->scal(&t, vec_V_, memspace_);
      // initialize norm history
      h_rs_[0] = rnorm;
      i = -1;
      inner_flag = 1;

      while((inner_flag) && (i + 1 < restart_)) {
        i++;
        it++;

        // Z_i = B * V_i
        vec_v->setData( vec_V_->getData(i, memspace_), memspace_);
  
        vec_z->setData( vec_Z_->getData(0, memspace_), memspace_);
        matrix_handler_->matvec(B_, vec_v, vec_z, &ONE, &ZERO, memspace_);
        
        mem_.deviceSynchronize();

        // V_{i+1}=A*Z_i

        vec_v->setData( vec_V_->getData(i + 1, memspace_), memspace_);

        matrix_handler_->matvec(A_, vec_z, vec_v, &ONE, &ZERO, memspace_);

        // orthogonalize V[i+1], form a column of h_H_

        GS_->orthogonalize(n_, vec_V_, h_H_, i);

        // Increment H_n_
        H_n_++;
        if (H_n_ > restart_) {
          H_n_ = 1;
        }

        // Givens Rotations on Previous Rows
        if (i != 0) {
          for (index_type k = 1; k <= i; k++) {
            k1 = k - 1;
            t = h_H_[i * (restart_ + 1) + k1];
            h_H_[i * (restart_ + 1) + k1] = h_c_[k1] * t + h_s_[k1] * h_H_[i * (restart_ + 1) + k];
            h_H_[i * (restart_ + 1) + k] = -h_s_[k1] * t + h_c_[k1] * h_H_[i * (restart_ + 1) + k];
          }
        } // if i!=0
        real_type Hii = h_H_[i * (restart_ + 1) + i];
        real_type Hii1 = h_H_[(i) * (restart_ + 1) + i + 1];
        real_type gam = std::sqrt(Hii * Hii + Hii1 * Hii1);

        if(std::abs(gam - ZERO) <= MACHINE_EPSILON) {
          gam = MACHINE_EPSILON;
        }

        /* next Given's rotation */
        h_c_[i] = Hii / gam;
        h_s_[i] = Hii1 / gam;
        h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
        h_rs_[i] = h_c_[i] * h_rs_[i];

        h_H_[(i) * (restart_ + 1) + (i)]     = h_c_[i] * Hii  + h_s_[i] * Hii1;
        h_H_[(i) * (restart_ + 1) + (i + 1)] = h_c_[i] * Hii1 - h_s_[i] * Hii;

         // Obtain parameter
        RS_ ->gcv(h_H_, h_rs_, &reg_param, H_n_, restart_, 1e-12, 100);

        // Call regularization solver
        RS_-> regularize(h_H_, h_rs_, y, reg_param, H_n_, restart_);

        // get solution
        vec_Z_->setToZero(memspace_);
        vec_z->setData( vec_Z_->getData(0, memspace_), memspace_);
        for (j = 0; j <= i; j++) {
          vec_v->setData( vec_V_->getData(j, memspace_), memspace_);
          vector_handler_->axpy(&y[j], vec_v, vec_z, memspace_);
        }

        // now multiply d_Z by B
        matrix_handler_->matvec(B_, vec_z, vec_x_i, &ONE, &ZERO, memspace_);

        // and add to x
        vector_handler_->axpy(&ONE, x, vec_x_i, memspace_);

        vec_norm->copyDataFrom(rhs, memspace_, memspace_);
        matrix_handler_->matvec(A_, vec_x_i, vec_norm, &MINUS_ONE, &ONE, memspace_);
        rnorm = vector_handler_->dot(vec_norm, vec_norm, memspace_);
        // rnorm = ||V_1||
        rnorm = std::sqrt(rnorm);

        // Relative Error
        vec_rel_error->copyDataFrom(x_true, memspace_, memspace_);
        vector_handler_->axpy(&MINUS_ONE, vec_x_i, vec_rel_error, memspace_);

        rel_error_norm_ = vector_handler_->dot(vec_rel_error, vec_rel_error, memspace_);
        rel_error_norm_ = std::sqrt(rnorm);
        rel_error_norm_ = rel_error_norm_/x_true_norm_;

        // File writing 
        file << i + 1 << "," << rnorm << "," << rel_error_norm_ << "," << reg_param << "\n";

        // residual norm estimate
        io::Logger::misc() << "it: " << it << " --> norm of the residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";
        // check convergence
        if (i + 1 >= restart_ || !stop_flag || it >= maxit_) {
          inner_flag = 0;
        }
      } // inner while

      io::Logger::misc() << "End of cycle, ESTIMATED norm of residual "
                         << std::scientific << std::setprecision(16)
                         << rnorm << "\n";

      /* test solution */

      if(!stop_flag || it >= maxit_) {
        outer_flag = 0;
        x->copyDataFrom(vec_x_i, memspace_, memspace_);
      }

      if(!outer_flag) {
        final_residual_norm_ = rnorm;
        total_iters_ = it;
        io::Logger::misc() << "End of cycle, COMPUTED norm of residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";
      }
      if(outer_flag){
        rhs->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);
        matrix_handler_->matvec(A_, vec_x_i, vec_V_, &MINUS_ONE, &ONE, memspace_);
        rnorm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
        // rnorm = ||V_1||
        rnorm = std::sqrt(rnorm);
      }
    } // outer while
    file.close();
    return 0;
  }

  int  LinSolverIterativeHybridABGMRES::resetMatrix(matrix::Sparse* new_matrix)
  {
    A_ = new_matrix;
    matrix_handler_->setValuesChanged(true, memspace_);
    return 0;
  }

  int  LinSolverIterativeHybridABGMRES::setupPreconditioner(std::string type, LinSolverDirect* LU_solver)
  {
    if (type != "LU") {
      out::warning() << "Only LU-type solve can be used as a preconditioner at this time." << std::endl;
      return 1;
    } else {
      LU_solver_ = LU_solver;
      return 0;
    }

  }

  /**
   * @brief Sets pointer to Gram-Schmidt (re)orthogonalization.
   *
   * @param[in] gs - pointer to Gram-Schmidt class instance.
   * @return 0 if successful, error code otherwise.
   */
  int LinSolverIterativeHybridABGMRES::setOrthogonalization(GramSchmidt* gs)
  {
    GS_ = gs;
    return 0;
  }

  /**
   * @brief Set/change GMRES restart value
   *
   * This function should leave solver instance in the same state but with
   * the new restart value.
   *
   * @param[in] restart - the restart value
   * @return 0 if successful, error code otherwise.
   *
   * @todo Consider not setting up GS, if it was not previously set up.
   */
  int LinSolverIterativeHybridABGMRES::setRestart(index_type restart)
  {
    // If the new restart value is the same as the old, do nothing.
    if (restart_ == restart) {
      return 0;
    }

    // Otherwise, set new restart value
    restart_ = restart;

    // If solver is already set, reallocate solver data
    if (is_solver_set_) {
      freeSolverData();
      allocateSolverData();
    }

    matrix_handler_->setValuesChanged(true, memspace_);

    // If Gram-Schmidt is already set, we need to reallocate it since the
    // restart value has changed.
    // if (GS_->isSetupComplete()) {
      GS_->setup(n_, restart_);
    // }

    return 0;
  }

  /**
   * @brief Set the convergence condition for GMRES solver
   *
   * @param[in] conv_cond - Possible values: 0, 1, 2
   * @return int - error code, 0 if successful
   */
  int LinSolverIterativeHybridABGMRES::setConvergenceCondition(index_type conv_cond)
  {
    conv_cond_ = conv_cond;
    return 0;
  }

  index_type  LinSolverIterativeHybridABGMRES::getRestart() const
  {
    return restart_;
  }

  index_type  LinSolverIterativeHybridABGMRES::getConvCond() const
  {
    return conv_cond_;
  }

  int LinSolverIterativeHybridABGMRES::setCliParam(const std::string id, const std::string value)
  {
    switch (getParamId(id))
    {
      case TOL:
        setTol(atof(value.c_str()));
        break;
      case MAXIT:
        setMaxit(atoi(value.c_str()));
        break;
      case RESTART:
        setRestart(atoi(value.c_str()));
        break;
      case CONV_COND:
        setConvergenceCondition(atoi(value.c_str()));
        break;
      default:
        std::cout << "Setting parameter failed!\n";
    }
    return 0;
  }

  std::string LinSolverIterativeHybridABGMRES::getCliParamString(const std::string id) const
  {
    switch (getParamId(id))
    {
      default:
        out::error() << "Trying to get unknown string parameter " << id << "\n";
    }
    return "";
  }

  index_type LinSolverIterativeHybridABGMRES::getCliParamInt(const std::string id) const
  {
    switch (getParamId(id))
    {
      case MAXIT:
        return getMaxit();
        break;
      case RESTART:
        return getRestart();
        break;
      case CONV_COND:
        return getConvCond();
        break;
      default:
        out::error() << "Trying to get unknown integer parameter " << id << "\n";
    }
    return -1;
  }

  real_type LinSolverIterativeHybridABGMRES::getCliParamReal(const std::string id) const
  {
    switch (getParamId(id))
    {
      case TOL:
        return getTol();
        break;
      default:
        out::error() << "Trying to get unknown real parameter " << id << "\n";
    }
    return std::numeric_limits<real_type>::quiet_NaN();
  }

  bool LinSolverIterativeHybridABGMRES::getCliParamBool(const std::string id) const
  {
    
    out::error() << "Trying to get unknown boolean parameter " << id << "\n";
    
    return false;
  }

  int LinSolverIterativeHybridABGMRES::printCliParam(const std::string id) const
  {
    switch (getParamId(id))
    {
    case TOL:
      std::cout << getTol() << "\n";
      break;
    case MAXIT:
      std::cout << getMaxit() << "\n";
      break;
    case RESTART:
      std::cout << getRestart() << "\n";
      break;
    case CONV_COND:
      std::cout << getConvCond() << "\n";
      break;
    default:
      out::error() << "Trying to print unknown parameter " << id << "\n";
      return 1;
    }
    return 0;
  }

  //
  // Private methods
  //

  int LinSolverIterativeHybridABGMRES::allocateSolverData()
  {
    vec_V_ = new vector_type(n_, restart_ + 1);
    vec_V_->allocate(memspace_);
    
    vec_Z_ = new vector_type(n_);
    vec_Z_->allocate(memspace_);

    h_H_  = new real_type[restart_ * (restart_ + 1)]();
    h_c_  = new real_type[restart_];      // needed for givens
    h_s_  = new real_type[restart_];      // same
    h_rs_ = new real_type[restart_ + 1]();  // for residual norm history

    return 0;
  }

  int LinSolverIterativeHybridABGMRES::freeSolverData()
  {
    delete [] h_H_ ;
    delete [] h_c_ ;
    delete [] h_s_ ;
    delete [] h_rs_;
    delete vec_V_;
    delete vec_Z_;

    h_H_  = nullptr;
    h_c_  = nullptr;
    h_s_  = nullptr;
    h_rs_ = nullptr;
    vec_V_  = nullptr;
    vec_Z_  = nullptr;

    return 0;
  }

  void LinSolverIterativeHybridABGMRES::setMemorySpace()
  {
    bool is_matrix_handler_cuda = matrix_handler_->getIsCudaEnabled();
    bool is_matrix_handler_hip  = matrix_handler_->getIsHipEnabled();
    bool is_vector_handler_cuda = matrix_handler_->getIsCudaEnabled();
    bool is_vector_handler_hip  = matrix_handler_->getIsHipEnabled();

    if ((is_matrix_handler_cuda != is_vector_handler_cuda) ||
        (is_matrix_handler_hip  != is_vector_handler_hip )) {
      out::error() << "Matrix and vector handler backends are incompatible!\n";
    }

    if (is_matrix_handler_cuda || is_matrix_handler_hip) {
      memspace_ = memory::DEVICE;
    } else {
      memspace_ = memory::HOST;
    }
  }

  void LinSolverIterativeHybridABGMRES::initParamList()
  {
    params_list_["tol"]       = TOL;
    params_list_["maxit"]     = MAXIT;
    params_list_["restart"]   = RESTART;
    params_list_["conv_cond"] = CONV_COND;
  }

} // namespace
