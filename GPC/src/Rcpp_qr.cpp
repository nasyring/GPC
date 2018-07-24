// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "RcppArmadillo.h"
#include <RcppParallel.h>
#include <Rcpp.h>
#include <math.h>
using namespace RcppParallel;
using namespace Rcpp;
using namespace arma;
using namespace std;
// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <cmath>
#include <algorithm>

// generic function for kl_divergence
template <typename InputIterator1, typename InputIterator2>
inline double kl_divergence(InputIterator1 begin1, InputIterator1 end1, 
                            InputIterator2 begin2) {
  
   // value to return
   double rval = 0;
   
   // set iterators to beginning of ranges
   InputIterator1 it1 = begin1;
   InputIterator2 it2 = begin2;
   
   // for each input item
   while (it1 != end1) {
      
      // take the value and increment the iterator
      double d1 = *it1++;
      double d2 = *it2++;
      
      // accumulate if appropirate
      if (d1 > 0 && d2 > 0)
         rval += std::log(d1 / d2) * d1;
   }
   return rval;  
}

// helper function for taking the average of two numbers
inline double average(double val1, double val2) {
   return (val1 + val2) / 2;
}

struct JsDistance : public Worker {
   
   // input matrix to read from
   const RMatrix<double> mat;
   
   // output matrix to write to
   RMatrix<double> rmat;
   
   // initialize from Rcpp input and output matrixes (the RMatrix class
   // can be automatically converted to from the Rcpp matrix type)
   JsDistance(const NumericMatrix mat, NumericMatrix rmat)
      : mat(mat), rmat(rmat) {}
   
   // function call operator that work for the specified range (begin/end)
   void operator()(std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; i++) {
         for (std::size_t j = 0; j < i; j++) {
            
            // rows we will operate on
            RMatrix<double>::Row row1 = mat.row(i);
            RMatrix<double>::Row row2 = mat.row(j);
            
            // compute the average using std::tranform from the STL
            std::vector<double> avg(row1.length());
            std::transform(row1.begin(), row1.end(), // input range 1
                           row2.begin(),             // input range 2
                           avg.begin(),              // output range 
                           average);                 // function to apply
              
            // calculate divergences
            double d1 = kl_divergence(row1.begin(), row1.end(), avg.begin());
            double d2 = kl_divergence(row2.begin(), row2.end(), avg.begin());
               
            // write to output matrix
            rmat(i,j) = sqrt(.5 * (d1 + d2));
         }
      }
   }
};


// [[Rcpp::export]]
NumericMatrix rcpp_parallel_js_distance(NumericMatrix mat) {
  
   // allocate the matrix we will return
   NumericMatrix rmat(mat.nrow(), mat.nrow());

   // create the worker
   JsDistance jsDistance(mat, rmat);
     
   // call it with parallelFor
   parallelFor(0, mat.nrow(), jsDistance);

   return rmat;
}

// helper function for Gibbs sampling

inline double GibbsMCMC(RVector<double> nn, RMatrix<double> data, RMatrix<double> thetaboot,
	RVector<double> bootmean0, RVector<double> bootmean1, RMatrix<double> databoot,
	RVector<double> alpha, RVector<double> M_samp, RVector<double> w, std::size_t i) {
   	

	
	
	
	double cov_ind;
	int M = int(M_samp[0]);
	int n = int(nn[0]);
   	NumericVector theta0old(1,0.0);
	NumericVector theta0new(1,0.0);
	NumericVector theta1old(1,0.0);
	NumericVector theta1new(1,0.0);
	NumericVector loglikdiff(1,0.0);
	NumericVector r(1,0.0);
	NumericVector uu(1,0.0);
	NumericVector postsamples0(M,0.0);
	NumericVector postsamples1(M,0.0);
	NumericVector l0(1,0.0);
	NumericVector l1(1,0.0);
	NumericVector u0(1,0.0);
	NumericVector u1(1,0.0);
	theta0old = thetaboot(i,0);
	theta1old = thetaboot(i,1);
	
	for(int j=0; j<(M+100); j++) {
		theta0new(0) = R::rnorm(theta0old(0), 0.5);
		loglikdiff(0) = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff(0) = loglikdiff(0) -w[0] * fabs(databoot(k,2*i+1)-theta0new(0) - theta1old(0)*databoot(k,2*i)) + w[0] * fabs(databoot(k,2*i+1)-theta0old(0) - theta1old(0)*databoot(k,2*i)); 
		}
		r[0] = R::dnorm(theta0new(0), theta0old(0),.5, 0)/R::dnorm(theta0old(0),theta0new(0),.5, 0);
		loglikdiff(0) = loglikdiff(0) + log(r(0));
		loglikdiff(0) = fmin(std::exp(loglikdiff(0)), 1.0);
		uu[0] = R::runif(0.0,1.0);
      		if((uu(0) <= loglikdiff(0)) && (j>99)) {
			postsamples0(j-100) = theta0new(0);
			theta0old(0) = theta0new(0); 
      		}
		else if(j>99){
			postsamples0(j-100) = theta0old(0);	
		}
		theta1new[0] = R::rnorm(theta1old(0), 0.5);
		loglikdiff(0) = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff(0) = loglikdiff(0) -w[0] * fabs(databoot(k,2*i+1)-theta0old(0) - theta1new(0)*databoot(k,2*i)) + w[0] * fabs(databoot(k,2*i+1)-theta0old(0) - theta1old(0)*databoot(k,2*i)); 
		}
		r[0] = R::dnorm(theta1new(0), theta1old(0),.5, 0) / R::dnorm(theta1old(0),theta1new(0),.5, 0);
		loglikdiff(0) = loglikdiff(0) + log(r(0));
		loglikdiff(0) = fmin(std::exp(loglikdiff(0)), 1.0);
		uu[0] = R::runif(0.0,1.0);
      		if((uu(0) <= loglikdiff(0)) && (j>99)) {
			postsamples1(j-100) = theta1new(0);
			theta1old(0) = theta1new(0); 
      		}
		else if(j>99){
			postsamples1(j-100) = theta1old(0);	
		}
	}
	std::sort(postsamples0.begin(), postsamples0.end());
	std::sort(postsamples1.begin(), postsamples1.end());
	l0[0] = postsamples0(0.025*M);
	u0[0] = postsamples0(0.975*M);
	l1[0] = postsamples1(0.025*M);
	u1[0] = postsamples1(0.975*M);
	if ( (l1[0] < bootmean1[0]) && (u1[0] > bootmean1[0]) ){
		cov_ind = 1.0;
	} else {cov_ind = 0.0;}
	
	return cov_ind;
	
}

// [[Rcpp::export]]
Rcpp::List GibbsMCMC2(NumericVector nn, NumericMatrix data, NumericMatrix thetaboot,
	NumericVector bootmean0, NumericVector bootmean1, NumericVector alpha, NumericVector M_samp, NumericVector w) {
   	
	List result;
	int M = int(M_samp[0]);
	int n = int(nn[0]);
   	NumericVector theta0old(1,0.0);
	NumericVector theta0new(1,0.0);
	NumericVector theta1old(1,0.0);
	NumericVector theta1new(1,0.0);
	NumericVector loglikdiff(1,0.0);
	NumericVector r(1,0.0);
	NumericVector uu(1,0.0);
	NumericVector postsamples0(M,0.0);
	NumericVector postsamples1(M,0.0);
	NumericVector l0(1,0.0);
	NumericVector l1(1,0.0);
	NumericVector u0(1,0.0);
	NumericVector u1(1,0.0);
	theta0old = bootmean0;
	theta1old = bootmean1;
	
	for(int j=0; j<(M+100); j++) {
		theta0new(0) = R::rnorm(theta0old(0), 0.5);
		loglikdiff(0) = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff(0) = loglikdiff(0) -w[0] * fabs(data(k,1)-theta0new(0) - theta1old(0)*data(k,0)) + w[0] * fabs(data(k,1)-theta0old(0) - theta1old(0)*data(k,0)); 
		}
		r[0] = R::dnorm(theta0new(0), theta0old(0),.5, 0)/R::dnorm(theta0old(0),theta0new(0),.5, 0);
		loglikdiff(0) = loglikdiff(0) + log(r(0));
		loglikdiff(0) = fmin(std::exp(loglikdiff(0)), 1.0);
		uu[0] = R::runif(0.0,1.0);
      		if((uu(0) <= loglikdiff(0)) && (j>99)) {
			postsamples0(j-100) = theta0new(0);
			theta0old(0) = theta0new(0); 
      		}
		else if(j>99){
			postsamples0(j-100) = theta0old(0);	
		}
		theta1new[0] = R::rnorm(theta1old(0), 0.5);
		loglikdiff(0) = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff(0) = loglikdiff(0) -w[0] * fabs(data(k,1)-theta0old(0) - theta1new(0)*data(k,0)) + w[0] * fabs(data(k,1)-theta0old(0) - theta1old(0)*data(k,0)); 
		}
		r[0] = R::dnorm(theta1new(0), theta1old(0),.5, 0) / R::dnorm(theta1old(0),theta1new(0),.5, 0);
		loglikdiff(0) = loglikdiff(0) + log(r(0));
		loglikdiff(0) = fmin(std::exp(loglikdiff(0)), 1.0);
		uu[0] = R::runif(0.0,1.0);
      		if((uu(0) <= loglikdiff(0)) && (j>99)) {
			postsamples1(j-100) = theta1new(0);
			theta1old(0) = theta1new(0); 
      		}
		else if(j>99){
			postsamples1(j-100) = theta1old(0);	
		}
	}
	std::sort(postsamples0.begin(), postsamples0.end());
	std::sort(postsamples1.begin(), postsamples1.end());
	l0[0] = postsamples0(0.025*M);
	u0[0] = postsamples0(0.975*M);
	l1[0] = postsamples1(0.025*M);
	u1[0] = postsamples1(0.975*M);
	
	result = Rcpp::List::create(Rcpp::Named("l0") = l0[0],Rcpp::Named("u0") = u0[0],Rcpp::Named("l1") = l1[0],Rcpp::Named("u1") = u1[0]);

	return result;
}

struct GPC_qr_mcmc_parallel : public Worker {

	const RVector<double> nn;
	const RMatrix<double> data;
	const RMatrix<double> thetaboot;
	const RVector<double> bootmean0;
	const RVector<double> bootmean1;
	const RMatrix<double> databoot;
	const RVector<double> alpha;
	const RVector<double> M_samp;
	const RVector<double> B_resamp;
	const RVector<double> w;
	RVector<double> cover;

   // initialize with source and destination
   GPC_qr_mcmc_parallel(const NumericVector nn,	const NumericMatrix data, const NumericMatrix thetaboot,
	const NumericVector bootmean0, const NumericVector bootmean1, const NumericMatrix databoot,
	const NumericVector alpha, const NumericVector M_samp, const NumericVector B_resamp,
	const NumericVector w, NumericVector cover) 
			: nn(nn), data(data), thetaboot(thetaboot), bootmean0(bootmean0), bootmean1(bootmean1), databoot(databoot), alpha(alpha), M_samp(M_samp), B_resamp(B_resamp), w(w), cover(cover) {}   

   // operator
void operator()(std::size_t begin, std::size_t end) {
		for (std::size_t i = begin; i < end; i++) {
			cover[i] = GibbsMCMC(nn, data, thetaboot, bootmean0, bootmean1, databoot, alpha, M_samp, w, i);	
		}
	}
};

// [[Rcpp::export]]
NumericVector rcpp_parallel_qr(NumericVector nn, NumericMatrix data, NumericMatrix thetaboot, NumericVector bootmean0,
	NumericVector bootmean1, NumericMatrix databoot, NumericVector alpha, NumericVector M_samp, NumericVector B_resamp,
	NumericVector w) {
	
   int B = int(B_resamp[0]);
   // allocate the matrix we will return
   NumericVector cover(B,2.0); 

   // create the worker
   GPC_qr_mcmc_parallel gpcWorker(nn, data, thetaboot, bootmean0, bootmean1, databoot, alpha, M_samp, B_resamp, w, cover);
     
   // call it with parallelFor
   
   parallelFor(0, B, gpcWorker);

   return cover;
}





// [[Rcpp::export]]
Rcpp::List GPC_qr_parallel(SEXP & nn, SEXP & data, SEXP & theta_boot, SEXP & data_boot, SEXP & alpha, SEXP & M_samp, SEXP & B_resamp) {

RNGScope scp;
Rcpp::Function _GPC_rcpp_parallel_qr("rcpp_parallel_qr");
List result;
List finalsample;
double eps 			= 0.01; 
NumericVector nn_ = Rcpp::as<NumericVector>(nn);
NumericMatrix data_ = Rcpp::as<NumericMatrix>(data);
NumericMatrix thetaboot_ = Rcpp::as<NumericMatrix>(theta_boot);
NumericVector bootmean0(1,0.0);
NumericVector bootmean1(1,0.0);
NumericMatrix databoot_ = Rcpp::as<NumericMatrix>(data_boot);
NumericVector alpha_ = Rcpp::as<NumericVector>(alpha);
NumericVector M_samp_ = Rcpp::as<NumericVector>(M_samp);
NumericVector B_resamp_ = Rcpp::as<NumericVector>(B_resamp);
NumericVector w(1,0.5);
double diff;
bool go 			= TRUE;
int t				=1; 
double sumcover;
int B = int(B_resamp_[0]);
NumericVector cover;
	
for (int i=0; i<B; i++) {
	bootmean0[0] = bootmean0[0] + thetaboot_(i,0);
	bootmean1[0] = bootmean1[0] + thetaboot_(i,1);
}
bootmean0 = bootmean0/B;
bootmean1 = bootmean1/B;

while(go){	
cover = _GPC_rcpp_parallel_qr(nn_, data_, thetaboot_, bootmean0, bootmean1, databoot_, alpha_, M_samp_, B_resamp_, w);
sumcover = 0.0;
for(int s = 0; s<B; s++){sumcover = sumcover + cover(s);}
diff = (sumcover/B) - (1.0-alpha_[0]);
if(((abs(diff)<= eps)&&(diff>=0)) || t>16) {
   go = FALSE;
} else {
   t = t+1;
   w[0] = fmax(w[0] + (pow(1+t,-0.51)*diff),0.1);
} 
}

// Final sample

NumericVector M_final; M_final[0] = 2*M_samp_[0];
finalsample = GibbsMCMC2(nn_, data_, thetaboot_, bootmean0, bootmean1, alpha_, M_final, w);
	
result = Rcpp::List::create(Rcpp::Named("w") = w,Rcpp::Named("t") = t,Rcpp::Named("diff") = diff, Rcpp::Named("list_cis") = finalsample);
	
return result;
}




// [[Rcpp::export]]
Rcpp::List GPC_qr(SEXP & nn, SEXP & data, SEXP & theta_boot, SEXP & data_boot, SEXP & alpha, SEXP & M_samp, SEXP & B_resamp) { 

List result;
double aalpha 			= Rcpp::as<double>(alpha);			 			
int n				= Rcpp::as<int>(nn);
int B 				= Rcpp::as<int>(B_resamp);
double eps 			= 0.01; 
double w			= 0.5;
arma::mat thetaboot     	= Rcpp::as<arma::mat>(theta_boot);
arma::mat ddata			= Rcpp::as<arma::mat>(data);
arma::mat databoot 		= Rcpp::as<arma::mat>(data_boot);
int M				= Rcpp::as<int>(M_samp);
arma::colvec postsamples0	= arma::colvec(M);
arma::colvec postsamples1	= arma::colvec(M);
arma::colvec postsamples0f	= arma::colvec(2*M);
arma::colvec postsamples1f	= arma::colvec(2*M);
arma::colvec sort0		= arma::colvec(M);
arma::colvec sort1		= arma::colvec(M);
double l0;
double l1;
double u0;
double u1;
NumericVector theta0old;
NumericVector theta0new;
NumericVector theta1old;
NumericVector theta1new;
NumericVector loglikdiff;
arma::colvec r			= arma::colvec(1);r.fill(0.0);
arma::colvec uu 		= arma::colvec(1);
arma::colvec cover		= arma::colvec(B);
double diff;
bool go 			= TRUE;
int t				=1; 
arma::colvec bootmean0		= arma::colvec(1);
arma::colvec bootmean1		= arma::colvec(1);
double sumcover = 0.0;

for (int i=0; i<B; i++) {
	bootmean0 = bootmean0 + thetaboot(i,0);
	bootmean1 = bootmean1 + thetaboot(i,1);
}
bootmean0 = bootmean0/B;
bootmean1 = bootmean1/B;



while(go) {
for (int i=0; i<B; i++) {
	theta0old = thetaboot(i,0);
	theta1old = thetaboot(i,1);
	for(int j=0; j<(M+100); j++) {
		theta0new = Rcpp::rnorm(1, theta0old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * fabs(databoot(k,2*i+1)-theta0new[0] - theta1old[0]*databoot(k,2*i)) + w * fabs(databoot(k,2*i+1)-theta0old[0] - theta1old[0]*databoot(k,2*i)); 
		}
		r = Rcpp::dnorm(theta0new, theta0old[0],.5)/Rcpp::dnorm(theta0old,theta0new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples0(j-100) = theta0new[0];
			theta0old = theta0new; 
      		}
		else if(j>99){
			postsamples0(j-100) = theta0old[0];	
		}
		theta1new = Rcpp::rnorm(1, theta1old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * fabs(databoot(k,2*i+1)-theta0old[0] - theta1new[0]*databoot(k,2*i)) + w * fabs(databoot(k,2*i+1)-theta0old[0] - theta1old[0]*databoot(k,2*i)); 
		}
		r = Rcpp::dnorm(theta1new, theta1old[0],.5) / Rcpp::dnorm(theta1old,theta1new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples1(j-100) = theta1new[0];
			theta1old = theta1new; 
      		}
		else if(j>99){
			postsamples1(j-100) = theta1old[0];	
		}
	}
	sort0 = sort(postsamples0);
	sort1 = sort(postsamples1);
	l0 = sort0(0.025*M);
	u0 = sort0(0.975*M);
	l1 = sort1(0.025*M);
	u1 = sort1(0.975*M);
	if ( (l1 < bootmean1(0)) && (u1 > bootmean1(0)) ){
		cover(i) = 1.0;
	} else {cover(i) = 0.0;}
}
sumcover = 0.0;
for(int s = 0; s<B; s++){sumcover = sumcover + cover(s);}

diff = (sumcover/B) - (1.0-aalpha);
if(((abs(diff)<= eps)&&(diff>=0)) || t>16) {
	go = FALSE;
} else {
	t = t+1;
	w = w + (pow(1+t,-0.51)*diff);
}
}

theta0old[0] = bootmean0(0);
theta1old[0] = bootmean1(0);
for(int j=0; j<(2*M+1000); j++) {
		theta0new = Rcpp::rnorm(1, theta0old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * fabs(ddata(k,1)-theta0new[0] - theta1old[0]*ddata(k,0)) + w * fabs(ddata(k,1)-theta0old[0] - theta1old[0]*ddata(k,0)); 
		}
		r = Rcpp::dnorm(theta0new, theta0old[0],.5) / Rcpp::dnorm(theta0old,theta0new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples0f(j-1000) = theta0new[0];
			theta0old = theta0new; 
      		}
		else if(j>999){
			postsamples0f(j-1000) = theta0old[0];	
		}
		theta1new = Rcpp::rnorm(1, theta1old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * fabs(ddata(k,1)-theta0old[0] - theta1new[0]*ddata(k,0)) + w * fabs(ddata(k,1)-theta0old[0] - theta1old[0]*ddata(k,0)); 
		}
		r = Rcpp::dnorm(theta1new, theta1old[0],.5) / Rcpp::dnorm(theta1old,theta1new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples1f(j-1000) = theta1new[0];
			theta1old = theta1new; 
      		}
		else if(j>999){
			postsamples1f(j-1000) = theta1old[0];	
		}
	}
	sort0 = sort(postsamples0f);
	sort1 = sort(postsamples1f);
	l0 = sort0(0.025*2*M);
	u0 = sort0(0.975*2*M);
	l1 = sort1(0.025*2*M);
	u1 = sort1(0.975*2*M);


result = Rcpp::List::create(Rcpp::Named("l0") = l0,Rcpp::Named("u0") = u0,Rcpp::Named("l1") = l1,Rcpp::Named("u1") = u1,Rcpp::Named("w") = w,Rcpp::Named("t") = t,Rcpp::Named("diff") = diff);


return result;
}


// [[Rcpp::export]]

inline bool compare(std::array<double, 6> a, std::array<double, 6> b){
    return (a[5] < b[5]);
}
Rcpp::List GPC_linreg(SEXP & nn, SEXP & data, SEXP & theta_boot, SEXP & data_boot, SEXP & alpha, SEXP & M_samp, SEXP & B_resamp) { 

List result;
double aalpha 			= Rcpp::as<double>(alpha);			 			
int n				= Rcpp::as<int>(nn);
int B 				= Rcpp::as<int>(B_resamp);
double eps 			= 0.01; 
double w			= 1.1;
arma::mat thetaboot     	= Rcpp::as<arma::mat>(theta_boot);
arma::mat ddata			= Rcpp::as<arma::mat>(data);
arma::mat databoot 		= Rcpp::as<arma::mat>(data_boot);
int M				= Rcpp::as<int>(M_samp);
arma::colvec sort0		= arma::colvec(M);
arma::colvec sort1		= arma::colvec(M);
arma::colvec sort2		= arma::colvec(M);
arma::colvec sort3		= arma::colvec(M);
double low [4];
double hi [4];
double low_f [4];
double hi_f [4];
double low_f80 [4];
double hi_f80 [4];
double low_f90 [4];
double hi_f90 [4];
arma::colvec intvs	= arma::colvec(8);
arma::colvec intvs9080	= arma::colvec(16);
NumericVector theta0old;
NumericVector theta0new;
NumericVector theta1old;
NumericVector theta1new;
NumericVector theta2old;
NumericVector theta2new;
NumericVector theta3old;
NumericVector theta3new;
NumericVector theta4old;
NumericVector theta4new;
NumericVector loglikdiff;
NumericVector loglik;
NumericVector loglik_temp;
arma::colvec r			= arma::colvec(1);r.fill(0.0);
arma::colvec uu 		= arma::colvec(1);
arma::colvec cover		= arma::colvec(B);
double diff;
bool go 			= TRUE;
int t				=1; 
arma::colvec bootmean0		= arma::colvec(1);
arma::colvec bootmean1		= arma::colvec(1);
arma::colvec bootmean2		= arma::colvec(1);
arma::colvec bootmean3		= arma::colvec(1);
arma::colvec bootmean4		= arma::colvec(1);
double sumcover = 0.0;

for (int i=0; i<B; i++) {
	bootmean0 = bootmean0 + thetaboot(i,0);
	bootmean1 = bootmean1 + thetaboot(i,1);
	bootmean2 = bootmean2 + thetaboot(i,2);
	bootmean3 = bootmean3 + thetaboot(i,3);
}
bootmean0 = bootmean0/B;
bootmean1 = bootmean1/B;
bootmean2 = bootmean2/B;
bootmean3 = bootmean3/B;
bootmean4 = bootmean4/B;
std::array<std::array<double, 6>, 2000> mcmc_samps;
std::array<std::array<double, 6>, 4000> mcmc_samps_f;


while(go) {
for (int i=0; i<B; i++) {
	theta0old = thetaboot(i,0);
	theta1old = thetaboot(i,1);
	theta2old = thetaboot(i,2);
	theta3old = thetaboot(i,3);
	theta4old = thetaboot(i,4);
	for(int j=0; j<M; j++) {
		theta0new = Rcpp::rnorm(1, theta0old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w *(1/theta4old)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0new*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2) + w* (1/theta4old)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2); 
		}
		r = Rcpp::dnorm(theta0new, theta0old[0],.1)/Rcpp::dnorm(theta0old,theta0new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0])) {
			mcmc_samps[j][0] = theta0new[0];
			theta0old = theta0new; 
      		}else {
			mcmc_samps[j][0] = theta0old[0];	
		}
		theta1new = Rcpp::rnorm(1, theta1old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * (1/theta4old)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1new*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2) + w* (1/theta4old)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2); 
		}
		r = Rcpp::dnorm(theta1new, theta1old[0],.1) / Rcpp::dnorm(theta1old,theta1new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0])) {
			mcmc_samps[j][1] = theta1new[0];
			theta1old = theta1new; 
      		}else {
			mcmc_samps[j][1] = theta1old[0];	
		}
		theta2new = Rcpp::rnorm(1, theta2old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * (1/theta4old)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2new*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2) + w* (1/theta4old)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2); 
		}
		r = Rcpp::dnorm(theta2new, theta2old[0],.1) / Rcpp::dnorm(theta2old,theta2new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0])) {
			mcmc_samps[j][2] = theta2new[0];
			theta2old = theta2new; 
      		}else {
			mcmc_samps[j][2] = theta2old[0];	
		}
		theta3new = Rcpp::rnorm(1, theta3old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * (1/theta4old)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3new*ddata(databoot(k, i)-1,4),2) + w* (1/theta4old)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2); 
		}
		r = Rcpp::dnorm(theta3new, theta3old[0],.1) / Rcpp::dnorm(theta3old,theta3new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0])) {
			mcmc_samps[j][3] = theta3new[0];
			theta3old = theta3new; 
      		}else {
			mcmc_samps[j][3] = theta3old[0];	
		}
		theta4new = Rcpp::rnorm(1, theta4old[0], 0.1);
		if(theta4new[0]>0.0){
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w*0.5*(log(6.2832*theta4new) -log(6.2832*theta4old))-w * (1/theta4new)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2) + w* (1/theta4old)*0.5*pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2); 
		}		
		r = Rcpp::dnorm(theta4new, theta4old[0],.1) / Rcpp::dnorm(theta4old,theta4new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
		if((uu(0) <= loglikdiff[0])) {
			mcmc_samps[j][4] = theta4new[0];
			theta4old = theta4new; 
      		}else {
			mcmc_samps[j][4] = theta4old[0];	
		}
		}
		loglik[0] = 0.0;
		loglik_temp[0] = 0.0;
		for(int k=0; k<n; k++){
			loglik_temp = pow(ddata(databoot(k, i)-1,0)-theta0old*ddata(databoot(k, i)-1,1)-theta1old*ddata(databoot(k, i)-1,2)-theta2old*ddata(databoot(k, i)-1,3)-theta3old*ddata(databoot(k, i)-1,4),2);
			loglik[0] = loglik[0] +  -(w/2.0)*log(6.2832*theta4old[0]) - (w/2.0)*(1/theta4old[0])*loglik_temp[0];
		}
		mcmc_samps[j][5] = loglik[0];	
	}
	
	std::sort (mcmc_samps.begin(), mcmc_samps.end(), compare); 
	low[0] = bootmean0(0);low[1] = bootmean1(0);low[2] = bootmean2(0);low[3] = bootmean3(0);
	hi[0] = bootmean0(0);hi[1] = bootmean1(0);hi[2] = bootmean2(0);hi[3] = bootmean3(0);
	for(int j=int(M*0.05); j<M; j++) {
		low[0] = fmin(low[0], mcmc_samps[j][0]);
		low[1] = fmin(low[1], mcmc_samps[j][1]);
		low[2] = fmin(low[2], mcmc_samps[j][2]);
		low[3] = fmin(low[3], mcmc_samps[j][3]);
		hi[0] = fmax(hi[0], mcmc_samps[j][0]);
		hi[1] = fmax(hi[1], mcmc_samps[j][1]);
		hi[2] = fmax(hi[2], mcmc_samps[j][2]);
		hi[3] = fmax(hi[3], mcmc_samps[j][3]);
	}
	if ( (low[3] < bootmean3(0)) && (hi[3] > bootmean3(0)) && (low[2] < bootmean2(0)) && (hi[2] > bootmean2(0)) && (low[1] < bootmean1(0)) && (hi[1] > bootmean1(0))){
		cover(i) = 1.0;
	} else {cover(i) = 0.0;}
}
sumcover = 0.0;
for(int s = 0; s<B; s++){sumcover = sumcover + cover(s);}
diff = (sumcover/B) - (1.0-aalpha);
if(((abs(diff)<= eps)&&(diff>=0)) || t>5) {
	go = FALSE;
} else {
	t = t+1;
	w = fmax(w + (pow(1+t,-0.51)*2*diff),0.05);
}
}

theta0old[0] = bootmean0(0);
theta1old[0] = bootmean1(0);
theta2old[0] = bootmean2(0);
theta3old[0] = bootmean3(0);
theta4old[0] = bootmean4(0);
	for(int j=0; j<(2*M); j++) {
		theta0new = Rcpp::rnorm(1, theta0old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * (1/theta4old)*0.5*pow(ddata(k,0)-theta0new*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* (1/theta4old)*0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta0new, theta0old[0],.1)/Rcpp::dnorm(theta0old,theta0new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0])) {
			mcmc_samps_f[j][0] = theta0new[0];
			theta0old = theta0new; 
      		}else {
			mcmc_samps_f[j][0] = theta0old[0];	
		}
		theta1new = Rcpp::rnorm(1, theta1old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * (1/theta4old)*0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1new*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* (1/theta4old)*0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta1new, theta1old[0],.1) / Rcpp::dnorm(theta1old,theta1new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0])) {
			mcmc_samps_f[j][1] = theta1new[0];
			theta1old = theta1new; 
      		}else {
			mcmc_samps_f[j][1] = theta1old[0];	
		}
		theta2new = Rcpp::rnorm(1, theta2old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * (1/theta4old)*0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2new*ddata(k,3)-theta3old*ddata(k,4),2) + w* (1/theta4old)*0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta2new, theta2old[0],.1) / Rcpp::dnorm(theta2old,theta2new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0])) {
			mcmc_samps_f[j][2] = theta2new[0];
			theta2old = theta2new; 
      		}else {
			mcmc_samps_f[j][2] = theta2old[0];	
		}
		theta3new = Rcpp::rnorm(1, theta3old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * (1/theta4old)*0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3new*ddata(k,4),2) + w*(1/theta4old)* 0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta3new, theta3old[0],.1) / Rcpp::dnorm(theta3old,theta3new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) ) {
			mcmc_samps_f[j][3] = theta3new[0];
			theta3old = theta3new; 
      		}else {
			mcmc_samps_f[j][3] = theta3old[0];	
		}
		theta4new = Rcpp::rnorm(1, theta4old[0], 0.1);
		if(theta4new[0]>0.0){
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w*0.5*(log(6.2832*theta4new) -log(6.2832*theta4old))-w * (1/theta4new)*0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* (1/theta4old)*0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}		
		r = Rcpp::dnorm(theta4new, theta4old[0],.1) / Rcpp::dnorm(theta4old,theta4new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
		if((uu(0) <= loglikdiff[0])) {
			mcmc_samps[j][4] = theta4new[0];
			theta4old = theta4new; 
      		}else {
			mcmc_samps[j][4] = theta4old[0];	
		}
		}
		loglik[0] = 0.0;
		loglik_temp[0] = 0.0;
		for(int k=0; k<n; k++){
			loglik_temp = pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2);
			loglik[0] = loglik[0] +  -(w/2.0)*log(6.2832*theta4old[0]) - (w/2.0)*(1/theta4old[0])*loglik_temp[0];
		}
		mcmc_samps_f[j][5] = loglik[0];	
	}
	
	std::sort (mcmc_samps_f.begin(), mcmc_samps_f.end(), compare); 	
	low_f[0] = bootmean0(0);low_f[1] = bootmean1(0);low_f[2] = bootmean2(0);low_f[3] = bootmean3(0);
	hi_f[0] = bootmean0(0);hi_f[1] = bootmean1(0);hi_f[2] = bootmean2(0);hi_f[3] = bootmean3(0);
	low_f90[0] = low_f[0];
	low_f90[1] = low_f[1];
	low_f90[2] = low_f[2];
	low_f90[3] = low_f[3];
	hi_f90[0] = hi_f[0];
	hi_f90[1] = hi_f[1];
	hi_f90[2] = hi_f[2];
	hi_f90[3] = hi_f[3];
	low_f80[0] = low_f[0];
	low_f80[1] = low_f[1]; 
	low_f80[2] = low_f[2];
	low_f80[3] = low_f[3];
	hi_f80[0] = hi_f[0];
	hi_f80[1] = hi_f[1];
	hi_f80[2] = hi_f[2];
	hi_f80[3] = hi_f[3];
	for(int j=int(2*M*aalpha); j<(2*M); j++) {
		low_f[0] = fmin(low_f[0], mcmc_samps_f[j][0]);
		low_f[1] = fmin(low_f[1], mcmc_samps_f[j][1]);
		low_f[2] = fmin(low_f[2], mcmc_samps_f[j][2]);
		low_f[3] = fmin(low_f[3], mcmc_samps_f[j][3]);
		hi_f[0] = fmax(hi_f[0], mcmc_samps_f[j][0]);
		hi_f[1] = fmax(hi_f[1], mcmc_samps_f[j][1]);
		hi_f[2] = fmax(hi_f[2], mcmc_samps_f[j][2]);
		hi_f[3] = fmax(hi_f[3], mcmc_samps_f[j][3]);
	}
	for(int j=int(2*M*0.1); j<(2*M); j++) {
		low_f90[0] = fmin(low_f90[0], mcmc_samps_f[j][0]);
		low_f90[1] = fmin(low_f90[1], mcmc_samps_f[j][1]);
		low_f90[2] = fmin(low_f90[2], mcmc_samps_f[j][2]);
		low_f90[3] = fmin(low_f90[3], mcmc_samps_f[j][3]);
		hi_f90[0] = fmax(hi_f90[0], mcmc_samps_f[j][0]);
		hi_f90[1] = fmax(hi_f90[1], mcmc_samps_f[j][1]);
		hi_f90[2] = fmax(hi_f90[2], mcmc_samps_f[j][2]);
		hi_f90[3] = fmax(hi_f90[3], mcmc_samps_f[j][3]);
	}
	for(int j=int(2*M*0.2); j<(2*M); j++) {
		low_f80[0] = fmin(low_f80[0], mcmc_samps_f[j][0]);
		low_f80[1] = fmin(low_f80[1], mcmc_samps_f[j][1]);
		low_f80[2] = fmin(low_f80[2], mcmc_samps_f[j][2]);
		low_f80[3] = fmin(low_f80[3], mcmc_samps_f[j][3]);
		hi_f80[0] = fmax(hi_f80[0], mcmc_samps_f[j][0]);
		hi_f80[1] = fmax(hi_f80[1], mcmc_samps_f[j][1]);
		hi_f80[2] = fmax(hi_f80[2], mcmc_samps_f[j][2]);
		hi_f80[3] = fmax(hi_f80[3], mcmc_samps_f[j][3]);
	}
	intvs(0) = low_f[0];
	intvs(1) = hi_f[0];
	intvs(2) = low_f[1];
	intvs(3) = hi_f[1];
	intvs(4) = low_f[2];
	intvs(5) = hi_f[2];
	intvs(6) = low_f[3];
	intvs(7) = hi_f[3];
	
	intvs9080(0) = low_f90[0];
	intvs9080(1) = hi_f90[0];
	intvs9080(2) = low_f90[1];
	intvs9080(3) = hi_f90[1];
	intvs9080(4) = low_f90[2];
	intvs9080(5) = hi_f90[2];
	intvs9080(6) = low_f90[3];
	intvs9080(7) = hi_f90[3];
	intvs9080(8) = low_f80[0];
	intvs9080(9) = hi_f80[0];
	intvs9080(10) = low_f80[1];
	intvs9080(11) = hi_f80[1];
	intvs9080(12) = low_f80[2];
	intvs9080(13) = hi_f80[2];
	intvs9080(14) = low_f80[3];
	intvs9080(15) = hi_f80[3];
	

result = Rcpp::List::create(Rcpp::Named("intvs") = intvs, Rcpp::Named("intvs9080") = intvs9080,Rcpp::Named("w") = w, Rcpp::Named("t") = t, Rcpp::Named("diff") = diff);
return result;	
}


// [[Rcpp::export]]

Rcpp::List GPC_varmix(SEXP & nn, SEXP & data, SEXP & data_boot, SEXP & alpha, SEXP & B_resamp) { 
RNGScope scp;
List result;
NumericVector   normarg		= Rcpp::NumericVector(1);normarg.fill(0.0);
NumericVector   normarg2	= Rcpp::NumericVector(1);normarg2.fill(0.0);
NumericVector	ddata		= Rcpp::as<NumericVector>(data);
arma::mat ddata_boot 		= Rcpp::as<arma::mat>(data_boot);
double aalpha 			= Rcpp::as<double>(alpha);			 			
int n				= Rcpp::as<int>(nn);
int B 				= Rcpp::as<int>(B_resamp);
double eps 			= 0.01;
double tol 			= 0.001;
NumericVector label		= rbinom(n,1,0.5);
double n1			= sum(label);
double mu1			= 0.0;
double mu2			= 0.0;
double sd1 = 0.0;
double sd2 = 0.0;
double sigma2 = 0.0;
double mdata = 0.0;
NumericVector phi1		= Rcpp::NumericVector(n);
NumericVector phi2		= Rcpp::NumericVector(n);
NumericVector ddata1		= Rcpp::NumericVector(n);ddata1.fill(0.0);
NumericVector ddata2		= Rcpp::NumericVector(n);ddata2.fill(0.0);
for(int i=0; i<n; i++){
	if(label[i]==1){
		ddata1[i]=ddata[i];
	}else {
		ddata2[i]=ddata[i];
	}
}
mu1 = sum(ddata1)/n1;
mu2 = sum(ddata2)/(n-n1);
mdata = sum(ddata)/n;
for(int i=0; i<n; i++){
	sigma2 = sigma2 + pow(ddata[i]-mdata,2);
	if(label[i]==1){
		sd1 = sd1 + pow(ddata1[i] - mu1,2);
	}else {
		sd2 = sd2 + pow(ddata2[i] - mu2,2);
	}
}
sigma2 = sigma2/(n-1);
sd1 = sqrt(sd1/n1);
sd2 = sqrt(sd2/(n-n1));
phi1.fill(n1/n);
phi2.fill(1-(n1/n));

// ELBO computation

NumericVector elbo_old = Rcpp::NumericVector(1);elbo_old.fill(0.0);
NumericVector elbo_new = Rcpp::NumericVector(1);elbo_new.fill(0.0);
NumericVector f1 = Rcpp::NumericVector(1);f1.fill(0.0);
NumericVector f2 = Rcpp::NumericVector(1);f2.fill(0.0);
NumericVector f3 = Rcpp::NumericVector(1);f3.fill(0.0);
NumericVector f4 = Rcpp::NumericVector(1);f4.fill(0.0);
NumericVector f5 = Rcpp::NumericVector(1);f5.fill(0.0);
NumericVector seq = Rcpp::NumericVector(1800);seq.fill(0.0);
NumericVector seq2 = Rcpp::NumericVector(600);seq.fill(0.0);
seq[0] = -9.0;
seq2[0] = -3.0;
for(int i=1; i<1800; i++){
	seq[i] = seq[i-1]+0.01;
}
for(int i=0; i<1800; i++){
	normarg[0] =  seq[i];
	f1 = f1 + 0.01*log(dnorm(normarg,0.0,sqrt(sigma2)))*dnorm(normarg,mu1,sd1);
	f2 = f2 + 0.01*log(dnorm(normarg,0.0,sqrt(sigma2)))*dnorm(normarg,mu2,sd2);
}
for(int i=0; i < n; i++){
	if(label[i]==1){
		for(int j=0; j<1800; j++){
			normarg[0] = ddata[i];
			normarg2[0] = seq[j];
			f3 = f3 + 0.01*log(dnorm(normarg,mu1,1.0))*dnorm(normarg2,mu1,sd1)*phi1[i];
		}
	} else {
		for(int j=0; j<1800; j++){
			normarg[0] = ddata[i];
			normarg2[0] = seq[j];
			f3 = f3 + 0.01*log(dnorm(normarg,mu2,1.0))*dnorm(normarg2,mu2,sd2)*phi2[i];
		}
	}
}
for(int i=0; i<600; i++){
	normarg[0] = seq2[i];
	f4 = f4 + 0.01*log(dnorm(normarg,mu1,sd1))*dnorm(normarg,mu1,sd1);
	f5 = f5 + 0.01*log(dnorm(normarg,mu2,sd2))*dnorm(normarg,mu2,sd2);
}

elbo_old[0] = f1[0] + f2[0] + n*log(0.5) + f3[0] + (-sum(log(phi1))) + f4[0] + f5[0]; 

bool go = TRUE;
int t = 0;
NumericVector p1k	= Rcpp::NumericVector(n);
NumericVector p2k	= Rcpp::NumericVector(n);
double ddiff = 0.0;
while(go){
	for(int i = 0; i<n; i++){
		p1k[i] = exp(mu1*ddata[i]-0.5*(pow(sd1,2)+pow(mu1,2)));
		p2k[i] = exp(mu2*ddata[i]-0.5*(pow(sd2,2)+pow(mu2,2)));
		phi1[i] = p1k[i]/(p1k[i]+p2k[i]);
		phi2[i] = p2k[i]/(p1k[i]+p2k[i]);
	}
	mu1 = 0.0;
	mu2 = 0.0;
	for(int i = 0; i<n; i++){
		mu1 = mu1 + ddata[i]*phi1[i]/((1/sigma2) + sum(phi1));
		mu2 = mu2 + ddata[i]*phi2[i]/((1/sigma2) + sum(phi2));
	}	
	sd1 = sqrt(1/((1/sigma2) + sum(phi1)));
	sd2 = sqrt(1/((1/sigma2) + sum(phi2)));
	for(int i = 0; i<n; i++){
		if(phi1[i]>phi2[i]){
			label[i] = 0;
		} else {
			label[i] = 1;
		}
	}
	//New ELBO
	f1[0] = 0.0;
	f2[0] = 0.0;
	f3[0] = 0.0;
	f4[0] = 0.0;
	f5[0] = 0.0;
	for(int i=0; i<1800; i++){
		normarg[0] = seq[i];
		f1 = f1 + 0.01*log(dnorm(normarg,0.0,sqrt(sigma2)))*dnorm(normarg,mu1,sd1);
		f2 = f2 + 0.01*log(dnorm(normarg,0.0,sqrt(sigma2)))*dnorm(normarg,mu2,sd2);
	}
	for(int i=0; i < n; i++){
		if(label[i]==1){
			for(int j=0; j<1800; j++){
				normarg[0] = ddata[i];
				normarg2[0] = seq[j];
				f3 = f3 + 0.01*log(dnorm(normarg,mu1,1.0))*dnorm(normarg2,mu1,sd1)*phi1[i];
			}
		} else {
			for(int j=0; j<1800; j++){
				normarg[0] = ddata[i];
				normarg2[0] = seq[j];
				f3 = f3 + 0.01*log(dnorm(normarg,mu2,1.0))*dnorm(normarg2,mu2,sd2)*phi2[i];
			}
		}
	}
	for(int i=0; i<600; i++){
		normarg[0] = seq2[i]; 
		f4 = f4 + 0.01*log(dnorm(normarg,mu1,sd1))*dnorm(normarg,mu1,sd1);
		f5 = f5 + 0.01*log(dnorm(normarg,mu2,sd2))*dnorm(normarg,mu2,sd2);
	}
	elbo_new[0] = f1[0] + f2[0] + n*log(0.5) + f3[0] + (-sum(log(phi1))) + f4[0] + f5[0]; 
	ddiff = fabs(elbo_new[0] - elbo_old[0]);
	if(ddiff<tol || t>15){
		go = FALSE;
	}
	elbo_old[0] = elbo_new[0];
	t = t+1;
}

 
double orig_mean1; 
double orig_mean2; 
double orig_sd1; 
double orig_sd2;

if(mu1<mu2){
orig_mean1 = mu1;
orig_mean2 = mu2;
orig_sd1 = sd1;
orig_sd2 = sd2;
}else{
orig_mean1 = mu2;
orig_mean2 = mu1;
orig_sd1 = sd2;
orig_sd2 = sd1;
}

//Bootstrapping
NumericVector cvg = Rcpp::NumericVector(B);cvg.fill(0.0);
double sumcover = 0.0;
double m1 = mu1;double m2 = mu2; double s1 = sd1; double s2 = sd2;
double w = 0.83;
bool go2 = TRUE;
NumericVector bootmu1 = Rcpp::NumericVector(B);bootmu1.fill(0.0);
NumericVector bootmu2 = Rcpp::NumericVector(B);bootmu2.fill(0.0);
NumericVector bootsd1 = Rcpp::NumericVector(B);bootsd1.fill(0.0);
NumericVector bootsd2 = Rcpp::NumericVector(B);bootsd2.fill(0.0);
	for(int b=0; b<B; b++){
		go2=TRUE;
		for(int i = 0; i<n; i++){
			ddata[i] = ddata_boot(i,b);
		}
		label		= rbinom(n,1,0.5);
		n1				= sum(label);
		mu1			= 0.0;
		mu2			= 0.0;
		sd1 = 0.0;
		sd2 = 0.0;
		sigma2 = 0.0;
		mdata = 0.0;
		phi1.fill(0.0);
		phi2.fill(0.0);
		ddata1.fill(0.0);
		ddata2.fill(0.0);
		for(int i=0; i<n; i++){
			if(label[i]==1){
				ddata1[i]=ddata[i];
			}else {
				ddata2[i]=ddata[i];
			}
		}
		mu1 = sum(ddata1)/n1;
		mu2 = sum(ddata2)/(n-n1);
		mdata = sum(ddata)/n;
		for(int i=0; i<n; i++){
			sigma2 = sigma2 + pow(ddata[i]-mdata,2);
			if(label[i]==1){
				sd1 = sd1 + pow(ddata1[i] - mu1,2);
			}else {
				sd2 = sd2 + pow(ddata2[i] - mu2,2);
			}
		}
		sigma2 = sigma2/(n-1);
		sd1 = sqrt(sd1/n1);
		sd2 = sqrt(sd2/(n-n1));
		phi1.fill(n1/n);
		phi2.fill(1-(n1/n));
		
		//ELBO
		
		elbo_old[0] = 0.0; f1[0] = 0.0;f2[0] = 0.0;f3[0] = 0.0;f4[0] = 0.0;f5[0] = 0.0;
		for(int i=0; i<1800; i++){
			normarg[0] = seq[i];
			f1 = f1 + 0.01*log(dnorm(normarg,0.0,sqrt(sigma2)))*dnorm(normarg,mu1,sd1);
			f2 = f2 + 0.01*log(dnorm(normarg,0.0,sqrt(sigma2)))*dnorm(normarg,mu2,sd2);
		}
		for(int i=0; i < n; i++){
			if(label[i]==1){
				for(int j=0; j<1800; j++){
					normarg[0] = ddata[i];
					normarg2[0] = seq[j];
					f3 = f3 + 0.01*log(dnorm(normarg,mu1,1.0))*dnorm(normarg2,mu1,sd1)*phi1[i];
				}
			} else {
				for(int j=0; j<1800; j++){
					normarg[0] = ddata[i];
					normarg2[0] = seq[j];
					f3 = f3 + 0.01*log(dnorm(normarg,mu2,1.0))*dnorm(normarg2,mu2,sd2)*phi2[i];
				}
			}
		}
		for(int i=0; i<600; i++){
			normarg[0] = seq2[i];
			f4 = f4 + 0.01*log(dnorm(normarg,mu1,sd1))*dnorm(normarg,mu1,sd1);
			f5 = f5 + 0.01*log(dnorm(normarg,mu2,sd2))*dnorm(normarg,mu2,sd2);
		}
		elbo_old[0] = f1[0] + f2[0] + n*log(0.5) + f3[0] + (-sum(log(phi1))) + f4[0] + f5[0]; 
		t=0;
		while(go2){
			for(int i = 0; i<n; i++){
				p1k[i] = exp(mu1*ddata[i]-0.5*(pow(sd1,2)+pow(mu1,2)));
				p2k[i] = exp(mu2*ddata[i]-0.5*(pow(sd2,2)+pow(mu2,2)));
				phi1[i] = p1k[i]/(p1k[i]+p2k[i]);
				phi2[i] = p2k[i]/(p1k[i]+p2k[i]);
			}
			mu1 = 0.0;
			mu2 = 0.0;
			for(int i = 0; i<n; i++){
				mu1 = mu1 + ddata[i]*phi1[i]/((1/sigma2) + sum(phi1));
				mu2 = mu2 + ddata[i]*phi2[i]/((1/sigma2) + sum(phi2));
			}	
			sd1 = sqrt(1/((1/sigma2) + sum(phi1)));
			sd2 = sqrt(1/((1/sigma2) + sum(phi2)));
			for(int i = 0; i<n; i++){
				if(phi1[i]>phi2[i]){
					label[i] = 0;
				} else {
					label[i] = 1;
				}
			}
			//New ELBO
			ddiff = 0.0;
			f1[0] = 0.0;
			f2[0] = 0.0;
			f3[0] = 0.0;
			f4[0] = 0.0;
			f5[0] = 0.0;
			for(int i=0; i<1800; i++){
				normarg[0] = seq[i];
				f1 = f1 + 0.01*log(dnorm(normarg,0.0,sqrt(sigma2)))*dnorm(normarg,mu1,sd1);
				f2 = f2 + 0.01*log(dnorm(normarg,0.0,sqrt(sigma2)))*dnorm(normarg,mu2,sd2);
			}
			for(int i=0; i < n; i++){
				if(label[i]==1){
					for(int j=0; j<1800; j++){
						normarg[0] = ddata[i];
						normarg2[0] = seq[j];
						f3 = f3 + 0.01*log(dnorm(normarg,mu1,1.0))*dnorm(normarg2,mu1,sd1)*phi1[i];
					}
				} else {
					for(int j=0; j<1800; j++){
						normarg[0] = ddata[i];
						normarg2[0] = seq[j];
						f3 = f3 + 0.01*log(dnorm(normarg,mu2,1.0))*dnorm(normarg2,mu2,sd2)*phi2[i];
					}
				}
			}
			for(int i=0; i<600; i++){
				normarg[0] = seq2[i];
				f4 = f4 + 0.01*log(dnorm(normarg,mu1,sd1))*dnorm(normarg,mu1,sd1);
				f5 = f5 + 0.01*log(dnorm(normarg,mu2,sd2))*dnorm(normarg,mu2,sd2);
			}
			elbo_new[0] = f1[0] + f2[0] + n*log(0.5) + f3[0] + (-sum(log(phi1))) + f4[0] + f5[0]; 
			ddiff = fabs(elbo_new[0] - elbo_old[0]);
			if(ddiff<tol || t>15){
				go2 = FALSE;
			}
			elbo_old[0] = elbo_new[0];
			t = t+1;
		}
		m1 = mu1;m2=mu2;s1=sd1;s2=sd2;
		if(m1 > m2){mu1 = mu2; mu2 = m1; sd1=sd2; sd2 = s1;}
		bootmu1[b] = mu1; bootmu2[b] = mu2; bootsd1[b] = sd1; bootsd2[b] = sd2;
	}
	bool go1 = TRUE;
	int tt = 0;
	while(go1){
	cvg.fill(0.0);
	sumcover = 0.0;
	ddiff = 0.0;
	for(int b = 0; b< B; b++){
	if( (bootmu1[b]-1.96*bootsd1[b]*pow(w,-1) <= orig_mean1) && (bootmu1[b]+1.96*bootsd1[b]*pow(w,-1) >= orig_mean1) && (bootmu2[b]-1.96*bootsd2[b]*pow(w,-1) <= orig_mean2) && (bootmu2[b]+1.96*bootsd2[b]*pow(w,-1)>= orig_mean2)){cvg[b]=1.0;}
	sumcover = sumcover + cvg[b];
	}
	ddiff = sumcover/B - (1-aalpha);
	if(fabs(ddiff)<=eps || tt>10){ go1 = FALSE;} else {
		tt=tt+1;
		w = fmax(w + ddiff*pow(1+tt,-0.51),0.05);
	}
}

NumericVector fin_int1 = Rcpp::NumericVector(2);
NumericVector fin_int2 = Rcpp::NumericVector(2);
double fin_cov1=0.0;
double fin_cov2=0.0;

fin_int1[0] = orig_mean1 - 1.96*orig_sd1*pow(w,-1);
fin_int1[1] = orig_mean1 + 1.96*orig_sd1*pow(w,-1);
fin_int2[0] = orig_mean2 - 1.96*orig_sd2*pow(w,-1);
fin_int2[1] = orig_mean2 + 1.96*orig_sd2*pow(w,-1);

if(fin_int1[0]<=(-2) && fin_int1[1]>=(-2)){fin_cov1=1.0;}
if(fin_int2[0]<=(2) && fin_int2[1]>=(2)){fin_cov2=1.0;}


result = Rcpp::List::create(Rcpp::Named("intv1") = fin_int1,Rcpp::Named("intv2") = fin_int2,Rcpp::Named("cov1") = fin_cov1,Rcpp::Named("cov2") = fin_cov2,Rcpp::Named("w") = w,Rcpp::Named("diff") = ddiff,Rcpp::Named("tt") = tt,Rcpp::Named("mean1") = orig_mean1,Rcpp::Named("mean2") = orig_mean2, Rcpp::Named("sumcover") = sumcover,Rcpp::Named("bootmu1") = bootmu1,Rcpp::Named("bootsd1") = bootsd1);

return result;


}

/*****************************
Parallel VaR GPC
*****************************/

// helper function for Gibbs sampling

inline double GibbsMCMCVaR(RVector<double> nn, RVector<double> qq, RVector<double> data, RVector<double> thetaboot,
	RVector<double> bootmean, RMatrix<double> databoot,
	RVector<double> alpha, RVector<double> M_samp, RVector<double> w, std::size_t i) {
   	

	double cov_ind;
	int M = int(M_samp[0]);
	int n = int(nn[0]);
   	NumericVector thetaold(1,0.0);
	NumericVector thetanew(1,0.0);
	NumericVector loglikdiff(1,0.0);
	NumericVector r(1,0.0);
	NumericVector uu(1,0.0);
	NumericVector postsamples(M,0.0);
	NumericVector l(1,0.0);
	NumericVector u(1,0.0);
	thetaold = thetaboot[i];
	
	for(int j=0; j<(M+100); j++) {
		thetanew(0) = R::rnorm(thetaold(0), 0.5);
		loglikdiff(0) = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff(0) = loglikdiff(0) -w[0] * 0.5*(fabs(thetanew(0)-databoot(k,i))-fabs(thetaold(0)-databoot(k,i))); 
		}
		loglikdiff(0) = (1/n)*loglikdiff(0);
		loglikdiff(0) = loglikdiff(0) + 0.5*(1-2*qq[0])*(thetanew(0)-thetaold(0));
		r[0] = R::dnorm(thetanew(0), thetaold(0),.5, 0)/R::dnorm(thetaold(0),thetanew(0),.5, 0);
		loglikdiff(0) = loglikdiff(0) + log(r(0));
		loglikdiff(0) = fmin(std::exp(loglikdiff(0)), 1.0);
		uu[0] = R::runif(0.0,1.0);
      		if((uu(0) <= loglikdiff(0)) && (j>99)) {
			postsamples(j-100) = thetanew(0);
			thetaold(0) = thetanew(0); 
      		}
		else if(j>99){
			postsamples(j-100) = thetaold(0);	
		}
	}
	std::sort(postsamples.begin(), postsamples.end());
	u[0] = postsamples((1-0.5*alpha[0])*M);
	l[0] = postsamples((0.5*alpha[0])*M);
	if ( (l[0] < bootmean[0]) && (u[0] > bootmean[0]) ){
		cov_ind = 1.0;
	} else {cov_ind = 0.0;}
	
	return cov_ind;
	
}

// [[Rcpp::export]]
Rcpp::List GibbsMCMCVaR2(NumericVector nn, NumericVector qq, NumericVector data, NumericVector thetaboot,
	NumericVector bootmean, NumericMatrix databoot,
	NumericVector alpha, NumericVector M_samp, NumericVector w) {

	
        List result;
	double cov_ind;
	int M = int(M_samp[0]);
	int n = int(nn[0]);
   	NumericVector thetaold(1,0.0);
	NumericVector thetanew(1,0.0);
	NumericVector loglikdiff(1,0.0);
	NumericVector r(1,0.0);
	NumericVector uu(1,0.0);
	NumericVector postsamples(M,0.0);
	NumericVector l(1,0.0);
	NumericVector u(1,0.0);
	thetaold[0] = bootmean[0];
	
	for(int j=0; j<(M+100); j++) {
		thetanew(0) = R::rnorm(thetaold(0), 0.5);
		loglikdiff(0) = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff(0) = loglikdiff(0) -w[0] * 0.5*(fabs(thetanew(0)-databoot(k,i))-fabs(thetaold(0)-databoot(k,i))); 
		}
		loglikdiff(0) = (1/n)*loglikdiff(0);
		loglikdiff(0) = loglikdiff(0) + 0.5*(1-2*qq[0])*(thetanew(0)-thetaold(0));
		r[0] = R::dnorm(thetanew(0), thetaold(0),.5, 0)/R::dnorm(thetaold(0),thetanew(0),.5, 0);
		loglikdiff(0) = loglikdiff(0) + log(r(0));
		loglikdiff(0) = fmin(std::exp(loglikdiff(0)), 1.0);
		uu[0] = R::runif(0.0,1.0);
      		if((uu(0) <= loglikdiff(0)) && (j>99)) {
			postsamples(j-100) = thetanew(0);
			thetaold(0) = thetanew(0); 
      		}
		else if(j>99){
			postsamples(j-100) = thetaold(0);	
		}
	}
	std::sort(postsamples.begin(), postsamples.end());
	u[0] = postsamples((1-0.5*alpha[0])*M);
	l[0] = postsamples((0.5*alpha[0])*M);
	
	result = Rcpp::List::create(Rcpp::Named("l") = l[0],Rcpp::Named("u") = u[0]);

	return result;
}

struct GPC_var_mcmc_parallel : public Worker {

	const RVector<double> nn;
	const RVector<double> qq;
	const RVector<double> data;
	const RVector<double> thetaboot;
	const RVector<double> bootmean;
	const RMatrix<double> databoot;
	const RVector<double> alpha;
	const RVector<double> M_samp;
	const RVector<double> B_resamp;
	const RVector<double> w;
	RVector<double> cover;

   // initialize with source and destination
   GPC_var_mcmc_parallel(const NumericVector nn, const NumericVector qq, const NumericVector data, const NumericVector thetaboot,
	const NumericVector bootmean, const NumericMatrix databoot,
	const NumericVector alpha, const NumericVector M_samp, const NumericVector B_resamp,
	const NumericVector w, NumericVector cover) 
			: nn(nn), qq(qq), data(data), thetaboot(thetaboot), bootmean(bootmean), databoot(databoot), alpha(alpha), M_samp(M_samp), B_resamp(B_resamp), w(w), cover(cover) {}   

   // operator
void operator()(std::size_t begin, std::size_t end) {
		for (std::size_t i = begin; i < end; i++) {
			cover[i] = GibbsMCMCVaR(nn, qq, data, thetaboot, bootmean, databoot, alpha, M_samp, w, i);	
		}
	}
};

// [[Rcpp::export]]
NumericVector rcpp_parallel_var(NumericVector nn, NumericVector qq, NumericVector data, NumericVector thetaboot, NumericVector bootmean,
	NumericMatrix databoot, NumericVector alpha, NumericVector M_samp, NumericVector B_resamp,
	NumericVector w) {
	
   int B = int(B_resamp[0]);
   // allocate the matrix we will return
   NumericVector cover(B,2.0); 

   // create the worker
   GPC_var_mcmc_parallel gpcWorker(nn, qq, data, thetaboot, bootmean, databoot, alpha, M_samp, B_resamp, w, cover);
     
   // call it with parallelFor
   
   parallelFor(0, B, gpcWorker);

   return cover;
}





// [[Rcpp::export]]
Rcpp::List GPC_var_parallel(SEXP & nn, SEXP & qq, SEXP & data, SEXP & theta_boot, SEXP & data_boot, SEXP & alpha, SEXP & M_samp, SEXP & B_resamp) {

RNGScope scp;
Rcpp::Function _GPC_rcpp_parallel_var("rcpp_parallel_var");
List result;
List finalsample;
double eps 			= 0.01; 
NumericVector nn_ = Rcpp::as<NumericVector>(nn);
NumericVector qq_ = Rcpp::as<NumericVector>(qq);
NumericMatrix data_ = Rcpp::as<NumericVector>(data);
NumericMatrix thetaboot_ = Rcpp::as<NumericVector>(theta_boot);
NumericVector bootmean(1,0.0);
NumericMatrix databoot_ = Rcpp::as<NumericMatrix>(data_boot);
NumericVector alpha_ = Rcpp::as<NumericVector>(alpha);
NumericVector M_samp_ = Rcpp::as<NumericVector>(M_samp);
NumericVector B_resamp_ = Rcpp::as<NumericVector>(B_resamp);
NumericVector w(1,0.5);
double diff;
bool go 			= TRUE;
int t				=1; 
double sumcover;
int B = int(B_resamp_[0]);
NumericVector cover;
	
for (int i=0; i<B; i++) {
	bootmean[0] = bootmean[0] + thetaboot_(i);
}
bootmean = bootmean/B;

while(go){	
cover = _GPC_rcpp_parallel_var(nn_, qq_, data_, thetaboot_, bootmean, databoot_, alpha_, M_samp_, B_resamp_, w);
sumcover = 0.0;
for(int s = 0; s<B; s++){sumcover = sumcover + cover(s);}
diff = (sumcover/B) - (1.0-alpha_[0]);
if(((abs(diff)<= eps)&&(diff>=0)) || t>16) {
   go = FALSE;
} else {
   t = t+1;
   w[0] = fmax(w[0] + (pow(1+t,-0.51)*diff),0.1);
} 
}

// Final sample

NumericVector M_final; M_final[0] = 2*M_samp_[0];
finalsample = GibbsMCMCVaR2(nn_, qq_, data_, thetaboot_, bootmean, alpha_, M_final, w);
	
result = Rcpp::List::create(Rcpp::Named("w") = w,Rcpp::Named("t") = t,Rcpp::Named("diff") = diff, Rcpp::Named("list_cis") = finalsample);
	
return result;
}





