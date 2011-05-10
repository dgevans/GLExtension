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

arma::imat computeTerms(int dim); 

template <int d>
class linearpoly {
    //contains the terms associated with a linear polynomial of dimension d
    static arma::imat terms;
    
    //Specific to each instanciation
    arma::vec coef;
    arma::mat Ainv;//matrix to back out coeefficients from state.
    
public:
    
    linearpoly(arma::mat ainv, arma::vec f);
    //constructs the terms associated
    static void constructTerms(){terms = computeTerms(d);};
    static const arma::imat& getTermMatrix(){return terms;}
    
    double evaluate(arma::vec x);//evaluates the linear polynomial at (x_1,\ldots x_n)
    
    arma::rowvec Jacobian(arma::vec x);
    
    static arma::rowvec getTermVector(arma::vec x);
};



template <int d>
arma::rowvec linearpoly<d>::getTermVector(arma::vec x) {
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
linearpoly<d>::linearpoly(arma::mat ainv,arma::vec f):Ainv(ainv) {
    if (Ainv.n_rows != terms.n_rows) {
        std::cerr<<"Linear Poly: Ainv has the wrong number of elements"<<std::endl;
        exit(1);
    }
    coef = Ainv*f;
}


template <int d>
double linearpoly<d>::evaluate(arma::vec x) {
    arma::mat ret = getTermVector(x)*coef; 
    return ret(0);
}

template <int d>
arma::rowvec linearpoly<d>::Jacobian(arma::vec x) {
    return getTermVector(x)*Ainv;
}



#endif