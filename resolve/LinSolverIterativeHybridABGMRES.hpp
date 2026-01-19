/**
 * @file LinSolverIterativeHybridABGMRES.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @brief Declaration of LinSolverIterativeHybridABGMRES class
 * 
 */
#pragma once

#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>
#include "GramSchmidt.hpp"
#include <resolve/LinSolverDirect.hpp>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/regularization/RegularizationSolver.hpp>

namespace ReSolve 
{
  /**
   * @brief Hybrid (AB)GMRES solver
   * 
   * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
   * @author Jeffery Zhang (jefferyz@vt.edu)
   * 
   * @note MatrixHandler and VectorHandler objects are inherited from
   * LinSolver base class.
   */
  class LinSolverIterativeHybridABGMRES : public LinSolverIterative
  {
    using vector_type = vector::Vector;

    public:
      LinSolverIterativeHybridABGMRES(MatrixHandler* matrix_handler,
                               VectorHandler* vector_handler,
                               GramSchmidt*   gs);
      LinSolverIterativeHybridABGMRES(MatrixHandler* matrix_handler,
                               VectorHandler* vector_handler,
                               GramSchmidt*   gs,
                               RegularizationSolver* rs);
      LinSolverIterativeHybridABGMRES(index_type restart,
                               real_type  tol,
                               index_type maxit,
                               index_type conv_cond,
                               MatrixHandler* matrix_handler,
                               VectorHandler* vector_handler,
                               GramSchmidt*   gs);
      ~LinSolverIterativeHybridABGMRES();

      int solve(vector_type* rhs, vector_type* x) override;
      int solve(vector_type* rhs, vector_type* x, real_type delta);
      int solveExport(vector_type* rhs, vector_type* x, vector_type* x_true);

      int setup(matrix::Sparse* A, matrix::Sparse* B);
      int resetMatrix(matrix::Sparse* new_A) override; 
      int setupPreconditioner(std::string name, LinSolverDirect* LU_solver) override;
      int setOrthogonalization(GramSchmidt* gs) override;

      int setRestart(index_type restart);
      int setConvergenceCondition(index_type conv_cond);
      index_type getRestart() const;
      index_type getConvCond() const;

      int setCliParam(const std::string id, const std::string value) override;
      std::string getCliParamString(const std::string id) const override;
      index_type getCliParamInt(const std::string id) const override;
      real_type getCliParamReal(const std::string id) const override;
      bool getCliParamBool(const std::string id) const override;
      int printCliParam(const std::string id) const override;

    private:
      enum ParamaterIDs {TOL=0, MAXIT, RESTART, CONV_COND};

      index_type restart_{10};  ///< GMRES restart
      index_type conv_cond_{0}; ///< GMRES convergence condition

    private:
      int allocateSolverData();
      int freeSolverData();
      void setMemorySpace();
      void initParamList();

      memory::MemorySpace memspace_;

      vector_type* vec_V_{nullptr};
      vector_type* vec_Z_{nullptr};

      real_type* h_H_{nullptr};
      real_type* h_c_{nullptr};
      real_type* h_s_{nullptr};
      real_type* h_rs_{nullptr};

      GramSchmidt* GS_{nullptr};
      
      RegularizationSolver* RS_{nullptr};

      LinSolverDirect* LU_solver_{nullptr};
      index_type n_{0};
      bool is_solver_set_{false};

      matrix::Sparse* B_{nullptr};

      MemoryHandler mem_; ///< Device memory manager object
  };
}
