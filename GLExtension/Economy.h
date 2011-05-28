//
//  Economy.h
//  GLExtension
//
//  Created by David Evans on 5/16/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//
#include "Firm.h"
#include <boost/random/mersenne_twister.hpp>

class economy {
    firm F;
    
    //Parameters
    boost::mt19937 gen;
    double psi; //How much do the agents dislike labor
    double beta;
    double var_e; //Variance of the productivity shock
    double sigma_e;
    double var_u; //Variance of the money shock
    double sigma_u;
    double rho; //persistance of the money shock
    double gamma;//Elasticity of substitution
    
    //Menu cost process
    arma::vec kappa;
    arma::mat Pi_k;
    arma::mat cumPi_k;
    int n_k; //number of kappa states
    
    //Beliefs
    arma::vec c_reg;
    
    
    
public:
    economy(double param[],const arma::vec &kap, const arma::mat &pi_k, const arma::vec &reg);//Constructor
    arma::vec simulateSeries(int Tburn, int Tmax, int Nfirms, int seed);
    
    int drawKappa(int i);
    
    void setBeliefs(arma::vec creg);
};

inline void economy::setBeliefs(arma::vec creg)
{
    F.creg() = creg;
    F.solveBellman();
}