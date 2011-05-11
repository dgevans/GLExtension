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
    linearpoly(){};
    linearpoly(const arma::mat &ainv);
    linearpoly(const arma::mat &ainv, const arma::vec &f);
    
    void fit(const arma::vec &f){coef = Ainv*f;};
    //constructs the terms associated
    static void constructTerms(){terms = computeTerms(d);};
    
    static const arma::umat& getTermMatrix(){return terms;}
    
    double evaluate(const arma::vec &x);//evaluates the linear polynomial at (x_1,\ldots x_n)
    
    arma::rowvec Jacobian(const arma::vec &x);
    
    static arma::rowvec getTermVector(const arma::vec &x);
    
    static arma::rowvec getSlopeTermVector(const arma::vec &x, int k);
    
    double getSlope(const arma::vec &x, int k);
};



template <int d>
arma::rowvec linearpoly<d>::getTermVector(const arma::vec &x) {
    arma::rowvec linterms = arma::ones<arma::rowvec>(terms.n_rows);
    for (int i=0; i<terms.n_rows; i++) {
        for (int j=0; j<d; j++) {
            if(terms(i,j) == 1)
                linterms(i) *= x(j);
        }
    }
    return linterms;
}

template <int d>
arma::rowvec linearpoly<d>::getSlopeTermVector(const arma::vec &x, int k) {
    arma::rowvec linterms = arma::ones<arma::rowvec>(terms.n_rows);
    for (int i=0; i<terms.n_rows; i++) {
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


template <int d>
linearpoly<d>::linearpoly(const arma::mat &ainv,const arma::vec &f):Ainv(ainv) {
    if (Ainv.n_rows != terms.n_rows) {
        std::cerr<<"Linear Poly: Ainv has the wrong number of elements"<<std::endl;
        exit(1);
    }
    fit(f);
}

template <int d>
linearpoly<d>::linearpoly(const arma::mat &ainv):Ainv(ainv) {
    if (Ainv.n_rows != terms.n_rows) {
        std::cerr<<"Linear Poly: Ainv has the wrong number of elements"<<std::endl;
        exit(1);
    }
}


template <int d>
double linearpoly<d>::evaluate(const arma::vec &x) {
    arma::mat ret = getTermVector(x)*coef; 
    return ret(0);
}

template <int d>
double linearpoly<d>::getSlope(const arma::vec &x,int k) {
    arma::mat ret = getSlopeTermVector(x,k)*coef; 
    return ret(0);
}

template <int d>
arma::rowvec linearpoly<d>::Jacobian(const arma::vec &x) {
    return getTermVector(x)*Ainv;
}



#endif