#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

class LinearKalmanFilter {

    /**
     * @brief Construct a new Linear Kalman Filter object and set the initial state of the filter later
     * 
     * @param A State transition matrix
     * @param B Input to state change matrix
     * @param C States to measurements matrix
     * @param D Inputs to measurements matrix
     * @param Q Process noise covariance
     * @param R Measurement noise covariance
     */
  LinearKalmanFilter(const MatrixXf &A, const MatrixXf &B, const MatrixXf &C,
                     const MatrixXf &D, const MatrixXf &Q, const MatrixXf &R);

    /**
     * @brief Construct a new Linear Kalman Filter object
     * 
     * @param A State transition matrix
     * @param B Input to state change matrix
     * @param C States to measurements matrix
     * @param D Inputs to measurements matrix
     * @param Q Process noise covariance
     * @param R Measurement noise covariance
     * @param x_hat Initial state esitmate
     * @param P_hat Initial covariance estimate
     */
  LinearKalmanFilter(const MatrixXf &A, const MatrixXf &B, const MatrixXf &C,
                     const MatrixXf &D, const MatrixXf &Q, const MatrixXf &R,
                     const VectorXf &x_hat, const MatrixXf &P_hat);

  ~LinearKalmanFilter();

public:

  /**
   * @brief Computes one step of the Kalman filter and updates P_hat and x_hat
   *
   * @param u The input to the system at this time step
   * @param z The measurements from the system at this timestep
   */
  void step(VectorXf u, VectorXf z);

  void set_A(const MatrixXf &A) { this->A = A; }
  void set_B(const MatrixXf &B) { this->B = B; }
  void set_C(const MatrixXf &C) { this->C = C; }
  void set_D(const MatrixXf &D) { this->D = D; }

  void set_Q(const MatrixXf &Q) {
    assert(("Q and A must have the same dimensions",
            n == Q.cols() && n == Q.rows()));
    this->Q = Q;
  }
  void set_R(const MatrixXf &R) {
    assert(("C rows must be equal to R cols and rows",
            p == R.cols() && p == R.rows()));
    this->R = R;
  }

  MatrixXf get_A() { return this->A; };
  MatrixXf get_B() { return this->B; };
  MatrixXf get_C() { return this->C; };
  MatrixXf get_D() { return this->D; };
  MatrixXf get_Q() { return this->Q; };
  MatrixXf get_R() { return this->R; };

  void set_n_states(const int i) { this->n = i; };
  void set_n_inputs(const int o) { this->m = o; };
  void set_n_measure(const int m) { this->p = m; };

  int get_n_states() { return n; };
  int get_n_inputs() { return m; };
  int get_n_measure() { return p; };

  void set_initial_filter_states(const VectorXf &x_hat, const MatrixXf &P_hat) {
    assert(("State guess must have dimensions consumate with A",
            x_hat.size() == this->A.rows()));
    assert(("State covariance must be square with dimensions of state",
            P_hat.rows() == this->n && P_hat.cols() == this->n));

    this->x_hat = x_hat;
    this->P_hat = P_hat;
  }

  MatrixXf get_P_hat() { return this->P_hat; };
  VectorXf get_x_hat() { return this->x_hat; };

private:
  void predict(VectorXf u);
  void update(VectorXf z);

  MatrixXf A;
  MatrixXf B;
  MatrixXf C;
  MatrixXf D;
  MatrixXf Q;
  MatrixXf R;

  VectorXf x_hat;
  MatrixXf P_hat;

  int n, m, p;
};
