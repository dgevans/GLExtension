//
//  main.cpp
//  GLExtension
//
//  Created by David Evans on 5/9/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "linearpoly.h"

using namespace arma;

template <int d> arma::imat linearpoly<d>::terms;


int main (int argc, const char * argv[])
{
    vec x(2),y(2),temp(2);
    vec f(4);
    mat A(4,4);
    x<< 0.0 <<1.0<<endr;
    y<< 0.0 <<1.0<<endr;
    f<< 0.0 <<1.0<<2.0<<4.0<<endr;
    linearpoly<2>::constructTerms();
    for (int i = 0; i<2;i++){
        for(int j=0; j<2;j++){
            temp<<x(i)<<x(j)<<endr;
            A.row(2*i+j) = linearpoly<2>::getTermVector(temp);
        }
    }
    A.print("A:");
    linearpoly<2> fit(inv(A),f);
    temp<<0.5<<0.5;
    // insert code here...
    std::cout << fit.evaluate(temp);
    (fit.Jacobian(temp)).print("J:");
    return 0;
}



