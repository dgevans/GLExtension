//
//  Firm.cpp
//  GLExtension
//
//  Created by David Evans on 5/11/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include "Firm.h"
#include "libscl.h"
#include <math.h>
#include <map>
#include <time.h>
#include <omp.h>

const double firm::pbound;
const double firm::maxTol;
template <int d> arma::umat linearpoly<d>::terms;

using namespace arma;
using namespace std;

typedef pair<double,double> pdd;//pdd is a pair of two doubles for use of keeping tack of p and g.

/*
 *Construct firm.
 */
firm::firm(double param[],const arma::vec &kap, const arma::mat &pi_k, const arma::vec &reg)
:kappa(kap),Pi_k(pi_k), c_reg(reg), grid(d,zeros<vec>(1))
{
    //Set up parameters
    psi = param[0];
    beta = param[1];
    var_e = param[2];
    var_u = param[3];
    sigma_e = sqrt(var_e);
    sigma_u = sqrt(var_u);
    rho = param[4];
    gamma = param[5];
    n_k = kappa.n_cols;
    
    //Golden search constants
    alpha1 = (3-sqrt(5))/2;
    alpha2 = (sqrt(5)-1)/2;
    
    //Set up quadrature

    e_u.zeros(N_Qu);
    w_u.zeros(N_Qu);
    scl::realmat e_r;
    scl::realmat w_r;
    scl::realmat endp(2,1,0.0);
    //Get Hermite quadrature nodes and wieghts for e^{-x^2}
    scl::gaussq(4, N_Qu, 0.0, 0.0, 0, endp, e_r, w_r);
    for(int i = 0; i<N_Qu;i++)
    {
        //adjust for gaussian weights and nodes
        e_u(i) = e_r[i+1]*sqrt(2)*sigma_u;
        w_u(i) = w_r[i+1]/sqrt(M_PI);
    }
    e_e.zeros(N_Qe);
    w_e.zeros(N_Qe);
    scl::gaussq(4, N_Qe, 0.0, 0.0, 0, endp, e_r, w_r);
    for(int i = 0; i<N_Qe;i++)
    {
        //adjust for gaussian weights and nodes
        e_e(i) = e_r[i+1]*sqrt(2)*sigma_e;
        w_e(i) = w_r[i+1]/sqrt(M_PI);
    }

    //Setup grids
    //bounds on mu
    mu_min = exp(e_u.min())*gamma/(gamma-1)*.15;
    mu_max = exp(e_u.max())*gamma/(gamma-1)*2;
    //bounds on g_t
    double g_max = 3*(sigma_e)/sqrt(1-rho*rho);
    double g_min = -3*(sigma_e)/sqrt(1-rho*rho);
    //bounds on log p_{t-1}
    double p_max = pbound;
    double p_min = -pbound;
    
    grid[0] = linspace<vec>(mu_min,mu_max, N_mu);
    grid[1] = linspace<vec>(g_min,g_max,N_g);
    grid[2] = linspace<vec>(p_min,p_max,N_p);
    
    V_c = std::vector< linint<d> >(n_k,linint<d>(grid));
    V_nc = std::vector< linint<d> >(n_k,linint<d>(grid));
    EV = std::vector< linint<d> >(n_k,linint<d>(grid));
    g = std::vector< linint<d> >(n_k,linint<d>(grid));
    
    //Initialize Value functions
    int ns = V_c[0].length();
    double mu = gamma/(gamma-1);
    vec va = (mu-1)*pow(mu,-gamma)/(1-beta*exp((gamma-1)*var_e/2))*ones<vec>(ns,1);
    //Set up bellman's with va as values.
    for (int i =0; i<n_k; i++) {
        V_c[i].f() = va;
        V_c[i].fit();
    }
    for (int i =0; i<n_k; i++) {
        V_nc[i].f() = va;
        V_nc[i].fit();
    }
    for (int i =0; i<n_k; i++) {
        EV[i].f() = va;
        EV[i].fit();
    }
    
    
    NVc = V_c[0].length();
    NVnc = V_nc[0].length();
    NEV = EV[0].length();
    Ng = g[0].length();
    
}

/*
 *Solves the Bellman equations for the firm
 */
void firm::solveBellman()
{
    //Get number of values for each Bellman equation
    NVc = V_c[0].length();
    NVnc = V_nc[0].length();
    NEV = EV[0].length();
    int N = n_k*(NVc+NVnc+NEV);
    //Holds new value of f and Jacobian
    vec fnew(N),Bel(N),f(N);
    mat Jac(N,N);
    vec diff = ones<vec>(N);
    while (norm(diff,"inf") > 1e-6)
    {
        //Get value of mapping and associated Jacobian
        iterateBellmanVc(Bel, Jac);
        iterateBellmanVnc(Bel, Jac);
        iterateBellmanEV(Bel, Jac);
        //stores current fvec in f
        getf(f);
        //computes difference from zero
        diff = Bel -f;
        //Update Jac to include -f term
        Jac -= eye(N, N);
        //Newton's method
        fnew = f-solve(Jac,diff);
        //set new f
        setf(fnew);
        cout<<norm(diff,"inf")<<endl;
    }
    computePolicy();
}

/*
 *Computes value of map and the associated Jacobian when firm changes price
 */
void firm::iterateBellmanVc(vec &Bel, mat &Jac)
{


    //iterate over cost states
    for(int i =0; i<n_k; i++)
    {
        //keeps track of wher in Bel we are
        int offset = i*NVc;
        int coffset = (NVc+NVnc)*n_k+i*NEV;
        map<pdd, pdd> cache;
        //Iterate over nodes
        int j;
#pragma omp parallel for private(j) shared(offset,coffset)
        for(j =0;j<NVc;j++)
        {
            vec temp;
            map<pdd, pdd>::iterator it;
            //Stores the g_t and p_{t-1} associated with node [j] and kappa_i in x
            pdd x = pdd(V_c[i][j](1),V_c[i][j](2));
            //check if have allready cached value
#pragma omp critical
            {
            it = cache.find(x);
            if (it == cache.end()) {
                //If not find optimal Mu node doesn't depend on mu today
                pdd mu = findOptimalMu(x,i);
                it = cache.insert( std::pair<pdd, pdd>(x,mu) ).first;
            }}
            pdd test = (*it).second;
            temp<<((*it).second).first<<x.first<<x.second;
            Bel(offset+j) = ((*it).second).second;
            //Vc and Vnc don't affect this map
            vec Jacpoly;
            uvec index = EV[i].Jacobian(temp,Jacpoly);
            rowvec Jactemp = zeros<rowvec>(NEV);
            Jactemp.elem(index) = beta*Jacpoly;
            Jac.row(offset+j)= zeros<rowvec>(Jac.n_cols);
            Jac.submat(offset+j, coffset, offset+j, coffset+NEV-1) =Jactemp;
        }
    }
}

/*
 *Computes value of map and associated Jacobian when does not change price
 */
void firm::iterateBellmanVnc(vec &Bel, mat &Jac)
{
    //iterate over cost states
    for(int i=0; i<n_k;i++)
    {
        int offset = NVc*n_k+NVnc*i;
        int coffset = (NVc+NVnc)*n_k+i*NEV;
        //Iterate over Nodes
#pragma omp parallel for
        for(int j=0;j<NVnc;j++)
        {
            vec x = V_nc[i][j];
            //get price today from beliefs
            double p = exp(c_reg(0)+c_reg(1)*x(1)+c_reg(2)*x(2));
            double mu = x(0);
            //present gain from mu
            Bel(offset+j) = (mu-1)*pow(mu,-gamma)*pow(p,1-gamma);
            //Vc and Vnc do not affect this mat
            Jac.row(offset+j) = zeros<rowvec>(Jac.n_cols);

            //gain from future choices
            Bel(offset+j) += beta*EV[i].eval(x);
            //Jacobian from the EV[i] f
            vec Jacpoly;
            uvec index = EV[i].Jacobian(x,Jacpoly);
            rowvec Jactemp = zeros<rowvec>(NEV);
            Jactemp.elem(index) = beta*Jacpoly;
            Jac.row(offset+j)= zeros<rowvec>(Jac.n_cols);
            Jac.submat(offset+j, coffset, offset+j, coffset+NEV-1) =Jactemp;
        
        }
    }
}


/*
 *Computes value of map and associated Jacobian for EV
 */
void firm::iterateBellmanEV(vec &Bel, mat &Jac)
{
    //iterate over cost states
    for(int i=0; i<n_k;i++)
    {
        int offset = NVc*n_k+NVnc*n_k+NEV*i;
        
        //Iterate over Nodes
#pragma omp parallel for
        for(int j=0; j<NEV; j++)
        {
            //mu_t g_t log(p_{t-1})
            vec x = EV[i][j];
            //get price today
            double logp = c_reg(0)+c_reg(1)*x(1)+c_reg(2)*x(2);
            Bel(offset+j) = 0;
            Jac.row(offset+j) = zeros<rowvec>(Jac.n_cols);
            //Computes expected returns
            for(int k=0; k<n_k;k++)
            {
                rowvec JactempVc = zeros<rowvec>(NVc);
                rowvec JactempVnc = zeros<rowvec>(NVnc);
                //sum over epsilon nodes
                for(int e_i = 0; e_i < N_Qe; e_i++)
                {
                    //sum over u nodes
                    for(int u_i=0;u_i<N_Qu;u_i++)
                    {
                        //mu_{t+1} g_{t+1} log(p_t)
                        vec xprime = zeros<vec>(d);
                        xprime(2) = logp;//log(p_t)
                        xprime(1) = rho*x(1)+e_u(u_i);//g_{t+1}
                        xprime(0) = x(0)/(exp(xprime(1)+e_e(e_i)));//mu_t/(exp(g_{t+1}+\epsilon_{t+1}))
                        //at xprmie which is larger V_c or V_nc, use that value
                        double Vmax = max(V_c[k].eval(xprime),V_nc[k].eval(xprime));
                        //Add to Bellman
                        Bel(offset+j) += Pi_k(i,k)*Vmax*w_e(e_i)*w_u(u_i);
                        //Update Jacobian
                        vec Jacpoly;
                        uvec index;
                        if (Vmax == V_c[k].eval(xprime))
                        {
                            index = V_c[k].Jacobian(xprime, Jacpoly);
                            JactempVc.elem(index) += Pi_k(i,k)*w_e(e_i)*w_u(u_i)*Jacpoly;
                            //If changing price has larger value
                        }else{
                            index = V_nc[k].Jacobian(xprime, Jacpoly);
                            JactempVnc.elem(index) +=Pi_k(i,k)*w_e(e_i)*w_u(u_i)*Jacpoly;
                            //If not changing price has larger value
                        }
                    }
                }
                int coffset = k*NVc;
                Jac.submat(offset+j, coffset, offset+j, coffset+NVc-1) = JactempVc;
                coffset = n_k*NVc+k*NVnc;
                Jac.submat(offset+j, coffset, offset+j, coffset+NVnc-1) = JactempVnc;
            }
        }
    }
}

/*
 *Find the optimal mu given xin = (g_t,log p_{t-1}) and current fixed costs today
 */
pdd firm::findOptimalMu(pdd xin, int k)
{
    vec x;
    //setup
    x << 0 << xin.first <<xin.second<<endr;
    //initial bounds
    double alim = exp(-0.2)*gamma/(gamma-1);
    double blim = exp(0.2)*gamma/(gamma-1);
    double a = alim;
    double b = blim;
    double mu1,mu2,f1,f2;
    double d = b-a;
    
    while (d > maxTol) {
        //two new middle points
        mu1 = a +alpha1*d;
        mu2 = a +alpha2*d;
        //Get values at those points
        x(0) = mu1;
        f1 = getPresentValue(x, k);
        x(0) = mu2;
        f2 = getPresentValue(x, k);
        if (f1 >= f2) {
            //if f1 > f2 then max is in [a,mu2]
            b = mu2;
        }else
        {
            //if f1<f2 then max is in [mu1,b]
            a = mu1;
        }
        d = b-a;
    }
    x(0) = (a+b)/2;
    if((abs(alim-x(0)) < maxTol)||(abs(blim-x(0)) < maxTol))
    {
        cerr<< "Maximization converged to limit!"<<endl;
        exit(1);
    }
    double temp = getPresentValue(x,k)-kappa(k)*psi;
    return pdd(x(0),temp);
}

/*
 *Get's the value today of markup mu_t, g_t and log p_{t-1}
 */
double firm::getPresentValue(vec &x, int k)
{
    double p = exp(c_reg(0)+c_reg(1)*x(1)+c_reg(2)*x(2));
    double mu = x(0);
    double ret = (mu-1)*pow(mu,-gamma)*pow(p,1-gamma)+beta*EV[k].eval(x);
    return ret;
}

/*
 *Stores the f's in the value functions in a vector
 */
void firm::getf(arma::vec &f)
{
    int N = n_k*(NVc+NVnc+NEV);
    f.reshape(N, 1);
    int offset;
    for(int i=0;i < n_k;i++)
    {
        offset = i*NVc;
        f.rows(offset,offset+NVc-1)=V_c[i].f();
    }
    
    for(int i=0;i < n_k;i++)
    {
        offset = n_k*NVc+i*NVnc;
        f.rows(offset,offset+NVnc-1)=V_nc[i].f();
    }
    
    for(int i=0;i < n_k;i++)
    {
        offset = n_k*NVc+n_k*NVnc+i*NEV;
        f.rows(offset,offset+NEV-1)=EV[i].f();
    }
}


/*
 *Stores an f vector in the Value function
 */

void firm::setf(const vec &f)
{
    int offset;
    for(int i=0;i < n_k;i++)
    {
        offset = i*NVc;
        V_c[i].f()=f.rows(offset,offset+NVc-1);
        V_c[i].fit();
    }
    
    for(int i=0;i < n_k;i++)
    {
        offset = n_k*NVc+i*NVnc;
        V_nc[i].f()=f.rows(offset,offset+NVnc-1);
        V_nc[i].fit();
    }
    
    for(int i=0;i < n_k;i++)
    {
        offset = n_k*NVc+n_k*NVnc+i*NEV;
        EV[i].f()=f.rows(offset,offset+NEV-1);
        EV[i].fit();
    }
}




void firm::computePolicy()
{
    //iterate over cost states
    for(int i =0; i<n_k; i++)
    {
        map<pdd, pdd> cache;
        //Iterate over nodes
#pragma omp parallel for
        for(int j =0;j<NVc;j++)
        {
            vec temp;
            map<pdd, pdd>::iterator it;
            //Stores the g_t and p_{t-1} associated with node [j] and kappa_i in x
            pdd x = pdd(g[i][j](1),g[i][j](2));
            //check if have allready cached value
            it = cache.find(x);
            if (it == cache.end()) {
                //If not find optimal Mu node doesn't depend on mu today
                pdd mu = findOptimalMu(x,i);
                it = cache.insert( std::pair<pdd, pdd>(x,mu) ).first;
            }
            g[i].f(j) = (*it).second.first;
            
        }
        g[i].fit();
    }
}

/*
 *Computes mu' given x = (mu,g,p_{t-1}) and cost state i;
 */
double firm::getPolicy(vec x,int i) const
{
    if(V_c[i].eval(x) >= V_nc[i].eval(x))
    {
        return g[i].eval(x);
    }else
    {
        if(x(0)>mu_max || x(0)<mu_min)
            return g[i].eval(x);
        
        //return mu if does not change prices.
        return x(0);
    }
}








