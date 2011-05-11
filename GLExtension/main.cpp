//
//  main.cpp
//  GLExtension
//
//  Created by David Evans on 5/9/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "linearinterpolator.h"

using namespace arma;

template <int d> arma::umat linearpoly<d>::terms;


int main (int argc, const char * argv[])
{
    vec temp(2),x1(2),x2(2);
    std::vector<vec> xnodes(2);
    vec f(4);
    mat A(4,4);
    xnodes[0]<< 0.0 <<1.0<<endr;
    xnodes[1]<< 0.0 <<1.0<<endr;
    f<< 0.0 <<1.0<<2.0<<4.0<<endr;
    linearpoly<2>::constructTerms();
    linint<2> interp(xnodes);
    interp.setf(f);
    interp.fit();
    temp << 0 << 1.0<<endr;
    std::cout<<interp.Jacobian(temp)<<std::endl;
    std::cout<<interp.getSlope(temp, 1)<<std::endl;
    
    // insert code here...
    return 0;
}



