//
//  main.cpp
//  GLExtension
//
//  Created by David Evans on 5/9/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "Economy.h"
#include <math.h>
#include <time.h>

using namespace arma;

template <int d> arma::umat linearpoly<d>::terms;
const int firm::d;

void solveBeliefs(economy &econ);
void rouwenhorst(double rho, double var, double mu, int N, vec &kap, mat &Pi_k);
int seed;

int main (int argc, const char * argv[])
{
    // insert code here...
    double psi = 1;
    double rho = pow(0.61,0.25);
    double var_e = 0.0231*0.0231;
    double var_u = 0.0018*0.0018/( (1+pow(rho,6))+(1+pow(rho,4))*pow((1+rho),2)+(1+pow(rho,2))*pow((1+rho+rho*rho),2)+pow((1+rho+rho*rho+rho*rho*rho),2));
    double gamma = 3;
    double beta = pow(0.96,1.0/52.0)/exp(pow((gamma-1),2)*var_e/2);

    double param[6] = {psi,beta,var_e,var_u,rho,gamma};
    double kapmu = 0.01;
    double kaprho = 0.95;
    double kapvar = .003*.003;
    vec kap;
    mat pi_k;
    rouwenhorst(kaprho, kapvar, kapmu, 4, kap, pi_k);
    vec reg;
    if(reg.load("creg.mat"))
    {
        if(reg.n_rows != 3)
            reg << 0.015<<-0.5<< 0.91;
    }else{
        reg << 0.015<<-0.5<< 0.91;
    }
    linearpoly<firm::d>::constructTerms();
    reg.print();
    economy econ( param, kap, pi_k, reg);
    seed = clock();
    solveBeliefs(econ);
    //compute stationary distribution

    return 0;
}

void solveBeliefs(economy &econ)
{
    vec creg;
    int n = 0;
    while (n<10) {
        creg = econ.simulateSeries(100, 500, 500000, clock());
        creg.print("New Beliefs: ");
        econ.setBeliefs(creg);
        creg.save("creg.mat");
        n++;
    }
}


void rouwenhorst(double rho, double var, double mu, int N, vec &kap, mat &pi_k)
{
    //set up
    double p = (1+rho)/2;
    double q = p;
    double psi = sqrt(N-1)*sqrt(var);
    
    //Compute transition
    mat theta, thetanew;
    theta << p << 1-p<<endr
    << 1-q << q<<endr;
    for(int n = 3; n<=N; n++)
    {
        //Follow algorithm to compute transition
        thetanew = zeros<mat>(n, n);
        thetanew.submat(0, 0, n-2,n-2) += p*theta;
        thetanew.submat(0, 1, n-2, n-1) += (1-p)*theta;
        thetanew.submat(1, 0, n-1, n-2) += (1-q)*theta;
        thetanew.submat(1, 1, n-1, n-1) += q*theta;
        thetanew.rows(1,n-2) /= 2;
        theta = thetanew;
    }
    pi_k = theta;
    //get states
    kap = linspace<vec>(-psi, psi, N)+mu;
}


