#include "kalman_filter.hpp"

#include <assert.h>

LinearKalmanFilter::LinearKalmanFilter(const MatrixXf &A, const MatrixXf &B,
                                       const MatrixXf &C, const MatrixXf &D,
                                       const MatrixXf &Q, const MatrixXf &R) {

  assert(("A must be a square matrix", A.rows() == A.cols()));
  assert(("B rows must equal A columns", A.rows() == B.cols()));
  assert(("C cols must eual A rows", C.cols() == A.rows()));

  assert(("Q and A must have the same dimensions",
          A.cols() == Q.cols() && A.rows() == Q.rows()));
  assert(("C rows must be equal to R cols and rows",
          C.rows() == R.cols() && C.rows() == R.rows()));

  set_A(A);
  set_B(B);
  set_C(C);
  set_D(D);

  set_n_states(A.rows());
  set_n_measure(C.rows());
  set_n_inputs(B.cols());
}

LinearKalmanFilter::LinearKalmanFilter(const MatrixXf &A, const MatrixXf &B,
                                       const MatrixXf &C, const MatrixXf &D,
                                       const MatrixXf &Q, const MatrixXf &R,
                                       const VectorXf &x_hat,
                                       const MatrixXf &P_hat) {

  LinearKalmanFilter(A, B, C, D, Q, R);
  set_initial_filter_states(x_hat, P_hat);
}

void LinearKalmanFilter::predict(VectorXf u) {
    x_hat = A*x_hat + B*u; // Predict state change
    P_hat = A * P_hat * A.transpose() + Q; // Predict covariance change
}

void LinearKalmanFilter::update(VectorXf z){
    VectorXf y = z - C*x_hat; // Innovation pre residual fit
    MatrixXf S = C * P_hat * C.transpose() + R; // Innovation of covariance pre residual fit

    MatrixXf K = P_hat * C.transpose() * S.inverse(); // Optimal Kalman gain

    x_hat = x_hat + K*y; // Updated state estimate
    P_hat = P_hat - K * C * P_hat; // Updated covariance estimate
}

void LinearKalmanFilter::step(VectorXf u, VectorXf z){
    predict(u);
    update(z);
}
