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
  x_hat = A * x_hat + B * u;             // Predict state change
  P_hat = A * P_hat * A.transpose() + Q; // Predict covariance change
}

void LinearKalmanFilter::update(VectorXf z) {
  VectorXf y = z - C * x_hat; // Innovation pre residual fit
  MatrixXf S = C * P_hat * C.transpose() +
               R; // Innovation of covariance pre residual fit

  MatrixXf K = P_hat * C.transpose() * S.inverse(); // Optimal Kalman gain

  x_hat = x_hat + K * y;         // Updated state estimate
  P_hat = P_hat - K * C * P_hat; // Updated covariance estimate
}

void LinearKalmanFilter::step(VectorXf u, VectorXf z) {
  predict(u);
  update(z);
}

/**
 * @brief Construct a new Extended Kalman Filter object
 * 
 * @param f Function f(x, u, dt) state transition model
 * @param h Function h(x, dt) sensor model
 * @param F Function F(x, u, dt) Jacobian of state transistion
 * @param H Function H(x, dt) Jacobian of measurement model
 * @param Q Process noise covariance
 * @param R Measurement noise covariance
 * @param ix_hat Initial guess for state
 * @param iP_hat Initial guess for state covariance
 */
ExtendedKalmanFilter::ExtendedKalmanFilter(
    std::function<VectorXf(VectorXf, VectorXf, float)> f,
    std::function<VectorXf(VectorXf, float)> h,
    std::function<MatrixXf(VectorXf, VectorXf, float)> F,
    std::function<MatrixXf(VectorXf, float)> H, const MatrixXf &Q,
    const MatrixXf &R, const VectorXf &ix_hat, const MatrixXf &iP_hat)
    : f(f), h(h), F(F), H(H), Q(Q), R(R), x_hat(ix_hat), P_hat(iP_hat) {}


void ExtendedKalmanFilter::predict(VectorXf u, float dt){
    MatrixXf cF  = F(x_hat, u, dt);

    x_hat = f(x_hat, u, dt); // Standard state prediction using model with discreate time
    P_hat = cF * P_hat * cF.transpose() + Q; // Covariance update using Jacobian
}

void ExtendedKalmanFilter::update(VectorXf z, float dt){

    MatrixXf cH = H(z, dt);

    VectorXf y = z - h(x_hat, dt); // Measurement innovation
    MatrixXf S = cH * P_hat * cH.transpose() + R; // Covariance innovation

    MatrixXf K = P_hat * cH * S.inverse(); // Near optimal kalman gain

    x_hat = x_hat + K * y; // State update
    P_hat = P_hat - K * cH * P_hat;
}

/**
 * @brief Step the Kalman filter forward in time. Carries out both the predict and update phases of the filter.
 * 
 * @param u Command at this timestep
 * @param z Measurements at this timestep
 * @param dt Length of this timestep
 */
void ExtendedKalmanFilter::step(VectorXf u, VectorXf z, float dt){
    predict(u, dt);
    update(z, dt);
}
