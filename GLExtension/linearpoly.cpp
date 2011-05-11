//
//  linearpoly.cpp
//  GLextension
//
//  Created by David Evans on 5/9/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include "linearpoly.h"
using namespace arma;

umat computeTerms(int dim)
{
    int n = int(pow(2,dim));
    umat termret(n,dim);
    if(dim !=1){
        umat prevterms = computeTerms(dim-1);
        for(int i =0; i<prevterms.n_rows; i++)
        {
            termret.submat(2*i, 0,2*i, dim-2) = prevterms.row(i);
            termret(2*i,dim-1) = 0;
            termret.submat(2*i+1, 0,2*i+1, dim-2) = prevterms.row(i);
            termret(2*i+1,dim-1) =1;
        }
    }else
    {
        termret(0,0) = 0;
        termret(1,0) = 1;
    }
    return termret;
}


