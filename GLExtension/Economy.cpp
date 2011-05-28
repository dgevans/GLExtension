//
//  Economy.cpp
//  GLExtension
//
//  Created by David Evans on 5/16/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include "Economy.h"
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <omp.h>

using namespace arma;

economy::economy(double param[],const arma::vec &kap, const arma::mat &pi_k, const arma::vec &reg):F(param,kap,pi_k,reg),kappa(kap),Pi_k(pi_k), c_reg(reg)
{
    //Set up parameters
    psi = param[0];
    beta = param[1];
    var_e = param[2];
    var_u = param[3];
    sigma_e = sqrt(var_e);
    sigma_u = sqrt(var_u);
    cout<<sigma_u;
    rho = param[4];
    gamma = param[5];
    n_k = kappa.n_rows;
    cumPi_k = join_cols(zeros<vec>(n_k),cumsum(Pi_k,1));
    
    F.solveBellman();
}

/*
 *Simulate the economy Tmax number of periods for Nfirms firms.
 */
vec economy::simulateSeries(int Tburn, int Tmax, int Nfirms, int seed)
{
    gen.seed(seed);
    boost::normal_distribution<> ndist(0.0,1.0);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > ngen(gen, ndist);
    //Initialize;
    vec mu = gamma/(gamma-1)*ones<vec>(Nfirms,1);
    ivec kap = zeros<ivec>(Nfirms);
    vec Fchange = zeros<vec>(Tmax+1);
    vec M(Tmax+1);
    M(0) = 1;
    vec g(Tmax+1);
    g(0) = 0;
    vec p(Tmax+1);//really log p
    p(0) = 0;
    //Run Economy
    for (int t =1; t<Tmax+1; t++) {
        g(t) = rho*g(t-1)+ngen()*sigma_u;
        M(t) = M(t-1)+exp(g(t));
        double P;
        //remember private x when parallelizing.
#pragma omp parallel
        {
            vec x;
            x << 0.0 <<g(t) <<p(t-1)<<endr;
#pragma omp for
            for (int j =0; j<Nfirms; j++) {
                double ea;
#pragma omp critical
                ea = ngen()*sigma_e;
                //update mu
                x(0) = mu(j)/exp(g(t)+ea);
                //get new mu
                mu(j) = F.getPolicy(x, kap(j));
                if (mu(j) != x(0)) {
#pragma omp critical
                    Fchange(t)++;
                }
                //add to price
#pragma omp critical
                P += pow(mu(j),1-gamma);
                //get new kappa
                kap(j) = drawKappa(kap(j));
            }
        }
        Fchange(t) /=Nfirms;
        P /= Nfirms;
        //get log P
        p(t) = log(P)/(1-gamma);
        
    }
    
    int n = Tmax - Tburn-1;
    //Perform Regression
    vec y = p.rows(Tburn+2, Tmax);
    mat X = join_rows(g.rows(Tburn+2,Tmax), p.rows(Tburn+1,Tmax-1));
    X = join_rows(ones<vec>(n), X);
    cout<<"Average number of changes: "<<accu(Fchange)/(Tmax+1)<<endl;
    return solve(trans(X)*X,trans(X)*y);
}

int economy::drawKappa(int i)
{
    boost::uniform_real<> dist(0,1.0);
    boost::variate_generator<boost::mt19937&, boost::uniform_real<> > udist(gen, dist);
    double rand;
#pragma omp critical
    rand = udist();
    int j;
    for (j = n_k -1; j>=0; j--) {
        if (rand > cumPi_k(i,j)) 
            break;
    }
    return j;
}