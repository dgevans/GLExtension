//
//  main.cpp
//  GLExtension
//
//  Created by David Evans on 5/9/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "Firm.h"
#include "libscl.h"
#include <math.h>
#include <time.h>

using namespace arma;

template <int d> arma::umat linearpoly<d>::terms;
const int firm::d;


int main (int argc, const char * argv[])
{
    // insert code here...
    double psi = 1;
    double rho = pow(0.61,0.25);
    double var_e = 0.0231*0.0231;
    double var_u = sqrt(0.0018*0.0018/( (1+pow(rho,6))+(1+pow(rho,4))*pow((1+rho),2)+(1+pow(rho,2))*pow((1+rho+rho*rho),2)+pow((1+rho+rho*rho+rho*rho*rho),2)));
    double gamma = 3;
    double beta = pow(0.96,1.0/52.0)/exp(pow((gamma-1),2)*var_e/2);

    double param[6] = {psi,beta,var_e,var_u,rho,gamma};
    vec kap;
    kap << 0.0167<<endr;
    mat pi_k;
    pi_k << 1.0<<endr;
    vec reg;
    reg << gamma/(1-gamma)<<0.0<<0.0;
    
    linearpoly<firm::d>::constructTerms();
    firm F( param, kap, pi_k, reg);
    F.solveBellman();
    return 0;
}


