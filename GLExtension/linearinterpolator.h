//
//  linearinterpolator.h
//  GLExtension
//
//  Created by David Evans on 5/10/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//
#ifndef LINEARINTERPOLATOR_H_
#define LINEARINTERPOLATOR_H_
#include "armadillo"
#include "linearpoly.h"
#include <vector>


template <int d>
class linint {
    std::vector<arma::vec> Xnodes;//Holds the coordinates where the linear interpolator is evaluated
    std::vector<arma::vec> X; //Holds the nodes where the linear interpolator is evaluated
    
    arma::uvec::fixed<d> Nnodes; //holds the number of nodes for each coordinate
    arma::uvec::fixed<d+1> prodNnodes;//holds the product for the purposes of interations
    
    std::vector< linearpoly<d> > polys; //The linear polynomial interpolator associated with each node
    
    arma::vec fnodes;//the value of the interpolator at each of the nodes.
    
    void createX();//Creates a list of all the coordnites where the function will be evaluated
    void createPolys();//create the polynomials at each node.
    
    arma::uvec intToVec(int n);//converts integer to which nodes
    arma::uvec intToVecBoundry(int n);//converts integer to nodes 1 away from boundry
    int vecToInt(arma::uvec nodes);//given a list of nodes gives row of X
    arma::uvec findNode(const arma::vec &x);//finds the closes node to x.
    
public:
    linint(const std::vector<arma::vec> &xnodes);
    void setf(const arma::vec &f){fnodes = f;};
    void fit();
    double eval(const arma::vec &x);//evaluates at location vector x.
    
    arma::rowvec Jacobian(const arma::vec &x);//finds the Jacobian w/ respect to fnodes at x
    
    double& f(int i){return fnodes(i);};
    const arma::vec& operator[](int i) const {return X[i];};
    
    int length(){return prodNnodes[d];};
    
    double getSlope(const arma::vec &x, int i);
};


template <int d>
linint<d>::linint(const std::vector<arma::vec> &xnodes):Xnodes(xnodes){
    if (xnodes.size() != d) {
        std::cerr<<"Wrong dimension for interpolator!"<<std::endl;
    }
    prodNnodes(0) = 1;
    for(int i =0; i< d; i++)
    {
        Nnodes(i) = xnodes[i].n_rows;
        prodNnodes(i+1) = prodNnodes(i)*Nnodes(i);
        Xnodes[i] = arma::sort(xnodes[i]);
    }
    createX();
    fnodes.zeros(prodNnodes(d));
    createPolys();
}

template <int d>
void linint<d>::createX() {
    X = std::vector<arma::vec>(prodNnodes(d),arma::zeros<arma::vec>(d));
    arma::uvec nodes;
    for (int n =0; n<prodNnodes(d); n++) {
        nodes = intToVec(n);
        for(int i=0; i<d;i++)
        {
            X[n](i) = (Xnodes[i])(nodes(i));
        }
    }
}

template <int d>
void linint<d>::createPolys() {
    polys = std::vector<linearpoly<d> >(prodNnodes(d));
    const arma::umat& terms = linearpoly<d>::getTermMatrix();
    int nterms = terms.n_rows;
    for(int n =0;n<prodNnodes(d);n++)
    {
        arma::mat temp(nterms,nterms);
        arma::uvec ncoord = intToVecBoundry(n);
        for(int i=0; i<nterms; i++)
        {
            arma::uvec t1 = ncoord+arma::trans(terms.row(i));
            arma::vec t2 = X[vecToInt(t1)];
            temp.row(i) = linearpoly<d>::getTermVector(t2);
        }
        polys[n] = linearpoly<d>(arma::inv(temp));
    }
}

template<int d>
void linint<d>::fit()
{
    const arma::umat& terms = linearpoly<d>::getTermMatrix();
    int nterms = terms.n_rows;
    for(int n =0;n<prodNnodes(d);n++)
    {
        arma::vec temp(nterms,1);
        arma::uvec ncoord = intToVecBoundry(n);
        for(int i=0; i<nterms; i++)
        {
            temp(i) = fnodes(vecToInt(ncoord+arma::trans(terms.row(i))));
        }
        
        polys[n].fit(temp);
    }
}

template<int d>
double linint<d>::eval(const arma::vec &x)
{
    arma::uvec node = findNode(x);
    return polys[vecToInt(node)].evaluate(x);
} 

template<int d>
double linint<d>::getSlope(const arma::vec &x,int i)
{
    arma::uvec node = findNode(x);
    return polys[vecToInt(node)].getSlope(x,i);
} 

template<int d>
arma::rowvec linint<d>::Jacobian(const arma::vec &x)
{
    arma::uvec node = findNode(x);
    return polys[vecToInt(node)].Jacobian(x);
} 

template<int d>
arma::uvec linint<d>::findNode(const arma::vec &x)
{
    arma::uvec node(d);
    for(int i =0; i<d; i++)
    {
        arma::uvec temp = arma::find(Xnodes[i] <= x(i),1,"last");
        node(i) = temp(0);
    }
    return node;
}


                        
template <int d>
arma::uvec linint<d>::intToVec(int n) {
    arma::uvec ret(d);
    int temp = n;
    for(int i=d-1; i >=0; i--)
    {
        ret(i) = temp/prodNnodes(i);
        temp = temp % prodNnodes(i);
    }
    return ret;
    
}

template <int d>
arma::uvec linint<d>::intToVecBoundry(int n) {
    arma::uvec ret(d);
    int temp = n;
    for(int i=d-1; i >=0; i--)
    {
        ret(i) = temp/prodNnodes(i);
        temp = temp % prodNnodes(i);
        if(ret(i) == Nnodes(i)-1)
            ret(i) -= 1;
    }
    return ret;
    
}

template <int d>
int linint<d>::vecToInt(arma::uvec nodes) {
    int n= 0;
    for(int i=d-1; i >=0; i--)
    {
        n += prodNnodes(i)*nodes(i);
    }
    return n;
    
}
#endif