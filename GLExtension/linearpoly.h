//
//  linearpoly.h
//  GLextension
//
//  Created by David Evans on 5/9/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//
#ifndef LINEARPOLY_H_
#define LINEARPOLY_H_
#include "armadillo"

arma::umat computeTerms(int dim); 

template <int d>
class linearpoly {
    //contains the terms associated with a linear polynomial of dimension d
    static arma::umat terms;
    
    //Specific to each instanciation
    arma::vec coef;
    arma::mat Ainv;//matrix to back out coeefficients from state.
    
    
public:
    //Constructors
    linearpoly(){};
    //ainv is a matrix such that coefficients = ainv*f;
    linearpoly(const arma::mat &ainv);
    linearpoly(const arma::mat &ainv, const arma::vec &f);
    
    //construct coefficients from f given cached Ainv
    void fit(const arma::vec &f){coef = Ainv*f;};
    //constructs the terms associated with this dimension of interpolator [1 ,0] represents x
    //[1,1] represents xy
    static void constructTerms(){terms = computeTerms(d);};
    //returns the term matrix
    static const arma::umat& getTermMatrix(){return terms;}
    
    double evaluate(const arma::vec &x) const;//evaluates the linear polynomial at (x_1,\ldots x_n)
    
    //Returns the Jacobian associated with f vector
    arma::rowvec Jacobian(const arma::vec &x) const;
    //gets the vector of terms at a given point x
    static arma::rowvec getTermVector(const arma::vec &x) ;
    //gets the vector of terms that are used to compute slope
    static arma::rowvec getSlopeTermVector(const arma::vec &x, int k);
    //gets the slope in direction k at point x
    double getSlope(const arma::vec &x, int k);
};


/*
 *gets the vector of terms at a given point x
 */
template <int d>
arma::rowvec linearpoly<d>::getTermVector(const arma::vec &x)  {
    arma::rowvec linterms = arma::ones<arma::rowvec>(terms.n_rows);
    for (int i=0; i<terms.n_rows; i++) {
        for (int j=0; j<d; j++) {
            // term [1,0,\ldots,1] represents x_1^1*x_2^0*\ldots x_d^1
            if(terms(i,j) == 1)
                linterms(i) *= x(j);
        }
    }
    return linterms;
}
/*
 *gets the vector of terms at a given point x for slope at direction k
 */
template <int d>
arma::rowvec linearpoly<d>::getSlopeTermVector(const arma::vec &x, int k) {
    arma::rowvec linterms = arma::ones<arma::rowvec>(terms.n_rows);
    for (int i=0; i<terms.n_rows; i++) {
        //only count terms where exponenent of x_k was 1
        if(terms(i,k) ==0)
            linterms(i) = 0;
        else{
            for (int j=0; j<d; j++) {
                if( (terms(i,j) == 1)&& (j!= k) )
                    linterms(i) *= x(j);
            }
        }
    }
    return linterms;
}

/*
 *Constructor stors ainv
 */
template <int d>
linearpoly<d>::linearpoly(const arma::mat &ainv,const arma::vec &f):Ainv(ainv) {
    if (Ainv.n_rows != terms.n_rows) {
        std::cerr<<"Linear Poly: Ainv has the wrong number of elements"<<std::endl;
        exit(1);
    }
    fit(f);
}
/*
 *Constructor stors ainv
 */
template <int d>
linearpoly<d>::linearpoly(const arma::mat &ainv):Ainv(ainv) {
    if (Ainv.n_rows != terms.n_rows) {
        std::cerr<<"Linear Poly: Ainv has the wrong number of elements"<<std::endl;
        exit(1);
    }
}

/*
 * Evaluates at given point x
 */
template <int d>
double linearpoly<d>::evaluate(const arma::vec &x) const{
    arma::mat ret = getTermVector(x)*coef; 
    return ret(0);
}
/*
 * Evaluates slope in direction k at given point x
 */
template <int d>
double linearpoly<d>::getSlope(const arma::vec &x,int k) {
    arma::mat ret = getSlopeTermVector(x,k)*coef; 
    return ret(0);
}
/*
 *Computes the effect of the vector f on the value at point x
 */
template <int d>
arma::rowvec linearpoly<d>::Jacobian(const arma::vec &x) const{
    return getTermVector(x)*Ainv;
}



#endif
