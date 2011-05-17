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
    
    arma::uvec intToVec(int n) const;//converts integer to which nodes
    arma::uvec intToVecBoundry(int n) const;//converts integer to nodes 1 away from boundry
    int vecToInt(arma::uvec nodes) const;//given a list of nodes gives row of X
    arma::uvec findNode(const arma::vec &x) const;//finds the closes node to x.
    
public:
    //Constructors
    linint();
    linint(const std::vector<arma::vec> &xnodes);
    //Allows one to set the value of the function at the nodes
    void setf(const arma::vec &f){fnodes = f;};
    //first to current values at fnodes
    void fit();
    double eval(const arma::vec &x);//evaluates at location vector x.
    
    arma::rowvec Jacobian(const arma::vec &x) const;//finds the Jacobian w/ respect to fnodes at x
    
    //allows one to change f at node i
    double& f(int i){return fnodes(i);};
    
    //Allows one to replace fnodes
    arma::vec& f(){return fnodes;};
    
    //Gives coordinates of ith node
    const arma::vec& operator[](int i) const {return X[i];};
    
    //Gets the number of nodes
    int length() const{return prodNnodes[d];};

    //Gets the slope at direction i
    double getSlope(const arma::vec &x, int i);
    
    //Allows one to change the nodes
    void setX(const std::vector<arma::vec> &xnodes);
};

template <int d>
linint<d>::linint():Xnodes(d,arma::zeros<arma::vec>(1))
{
    
}

template <int d>
linint<d>::linint(const std::vector<arma::vec> &xnodes):Xnodes(xnodes){
    if (xnodes.size() != d) {
        std::cerr<<"Wrong dimension for interpolator!"<<std::endl;
    }
    //Prodnodes holds number of nodes to create entire vector of nodes
    prodNnodes(0) = 1;
    for(int i =0; i< d; i++)
    {
        Nnodes(i) = xnodes[i].n_rows;
        prodNnodes(i+1) = prodNnodes(i)*Nnodes(i);
        Xnodes[i] = arma::sort(xnodes[i]);
    }
    //creates vector of all the nodes
    createX();
    //itializes  fnodes to be zero
    fnodes.zeros(prodNnodes(d));
    //creates the polynomial interpolators
    createPolys();
}

template <int d>
void linint<d>::setX(const std::vector<arma::vec> &xnodes){
    if (xnodes.size() != d) {
        std::cerr<<"Wrong dimension for interpolator!"<<std::endl;
    }
    Xnodes = xnodes;
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
        //given i gives which nodes that i is associated to
        nodes = intToVec(n);
        for(int i=0; i<d;i++)
        {
            //store those nodes in X
            X[n](i) = (Xnodes[i])(nodes(i));
        }
    }
}

template <int d>
void linint<d>::createPolys() {
    //creates a polynomial at each node
    polys = std::vector<linearpoly<d> >(prodNnodes(d));
    //get the matrix of terms
    const arma::umat& terms = linearpoly<d>::getTermMatrix();
    int nterms = terms.n_rows;
    for(int n =0;n<prodNnodes(d);n++)
    {
        arma::mat temp(nterms,nterms);
        //gets nodes associated with i, but if on boundry moves one to interior
        arma::uvec ncoord = intToVecBoundry(n);
        for(int i=0; i<nterms; i++)
        {
            //value at each neighborhood node creates the fit
            arma::uvec t1 = ncoord+arma::trans(terms.row(i));
            arma::vec t2 = X[vecToInt(t1)];
            temp.row(i) = linearpoly<d>::getTermVector(t2);
        }
        //sets up polynomial
        polys[n] = linearpoly<d>(arma::inv(temp));
    }
}

/*
 *Creates coefficient for each polynomial at each node given fnode
 */
template<int d>
void linint<d>::fit()
{
    //term matrix allows us to get neighbors
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
arma::rowvec linint<d>::Jacobian(const arma::vec &x) const
{
    //Setup
    arma::rowvec Jac = arma::zeros<arma::rowvec>(length());
    const arma::umat& terms = linearpoly<d>::getTermMatrix();
    //Find the node corresponding to x
    int node = vecToInt(findNode(x));
    //Get the Jacobian in polyspace at x
    arma::rowvec polyJac = polys[node].Jacobian(x);
    //get the node for which the fit is made
    arma::uvec ncoord = intToVecBoundry(node);
    //convert jacobian to the interp space.
    int nterms = terms.n_rows;
    for(int i=0; i<nterms; i++)
    {
        Jac(vecToInt(ncoord+arma::trans(terms.row(i)))) = polyJac(i);
    }
    return Jac;
} 

template<int d>
arma::uvec linint<d>::findNode(const arma::vec &x) const
{
    arma::uvec node(d);
    for(int i =0; i<d; i++)
    {
        arma::uvec temp = arma::find(Xnodes[i] <= x(i),1,"last");
        if(temp.n_rows != 0 )
            node(i) = temp(0);
        else
            node(i) = 0;
    }
    return node;
}


                        
template <int d>
arma::uvec linint<d>::intToVec(int n) const {
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
arma::uvec linint<d>::intToVecBoundry(int n) const {
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
int linint<d>::vecToInt(arma::uvec nodes) const{
    int n= 0;
    for(int i=d-1; i >=0; i--)
    {
        n += prodNnodes(i)*nodes(i);
    }
    return n;
    
}
#endif