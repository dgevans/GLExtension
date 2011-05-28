//
//  Firm.h
//  GLExtension
//
//  Created by David Evans on 5/11/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//
#ifndef FIRM_H_
#define FIRM_H_
#include "linearinterpolator.h"

class firm {
public:
    //some constants
    static const int d = 3;//Size of interpolators used
    static const int N_Qu = 40;//number of nodes for quadrature for u shock.
    static const int N_Qe = 20;//number of nodes for quadrature for e shock.
    static const int N_mu = 50;
    static const int N_g = 7;
    static const int N_p = 7;
    static const double pbound = 0.015;
    static const double maxTol = 1e-7;//Tolerance form maximization
private:
    //things I don't want to compute many times
    //For golden search
    double alpha1;
    double alpha2;
    
    //Parameters
    double psi; //How much do the agents dislike labor
    double beta;
    double var_e; //Variance of the productivity shock
    double sigma_e;
    double var_u; //Variance of the money shock
    double sigma_u;
    double rho; //persistance of the money shock
    double gamma;//Elasticity of substitution
    double mu_min;
    double mu_max;
    
    //Menu cost process
    arma::vec kappa;
    arma::mat Pi_k;
    int n_k; //number of kappa states
    
    //Beliefs
    arma::vec c_reg;//coefficients of the regression equation predicting
    
    
    //Grids
    std::vector<arma::vec> grid;//keeps track of the grid points
    
    //Value functions
    std::vector< linint<d> >V_c;//value function if change mu
    int NVc;//Number of nodes in each interpolator of V_c
    std::vector< linint<d> >V_nc;//Value function if mu is not changed
    int NVnc;//Number of nodes in each interpolator of V_nc
    std::vector< linint<d> >EV;//Expected value tomorrow.
    int NEV;//Number of nodes in each interpolator of EV
    std::vector< linint<d> >g;//Policy function
    int Ng;
    
    //Gauss Quadrature nodes and weights
    arma::vec e_u;//nodes
    arma::vec w_u; //weights
    arma::vec e_e;
    arma::vec w_e;
    
    
    //methods
    void setf(const arma::vec &f);
    void getf(arma::vec &f);
    
    //methods to compute new f and associated Jacobian of matrix
    void iterateBellmanVc(arma::vec &Bel, arma::mat &Jac);
    void iterateBellmanVnc(arma::vec &Bel, arma::mat &Jac);
    void iterateBellmanEV(arma::vec &Bel, arma::mat &Jac);
    std::pair<double,double> findOptimalMu(std::pair<double,double> x, int k);//Find the optimal mu given x =(g_t,p_{t-1}), return <mu,value>
    double getPresentValue(arma::vec &x, int k);
    
    void computePolicy();
    
public:
    //Solves the Bellman equations
    void solveBellman();
    
    firm(double param[],const arma::vec &kap, const arma::mat &pi_k, const arma::vec &reg);
    
    double getPolicy(arma::vec x,int i) const;
    
    arma::vec& creg() {return c_reg;};
};










#endif