// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "RcppArmadillo.h"
#include <RcppParallel.h>
#include <Rcpp.h>
using namespace RcppParallel;
using namespace Rcpp;
using namespace arma;
using namespace std;
// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]





// [[Rcpp::export]]
struct GPC_qr_mcmc_parallel : public Worker
{
const int nn;
const arma::mat data;
const arma::mat thetaboot;
const arma::mat bootmean0;
const arma::mat bootmean1;
const arma::mat databoot;
const double alpha;
const int M_samp;
const int B_resamp;
const double w;
arma::colvec cover;

   // initialize with source and destination
   GPC_qr_mcmc_parallel(const int nn, const arma::mat data, const arma::mat thetaboot, const arma::mat bootmean0, const arma::mat bootmean1, const arma::mat databoot,
   			const double alpha, const int M_samp, const int B_resamp, const double w, arma::colvec cover) 
			: nn(nn), data(data), thetaboot(thetaboot), bootmean0(bootmean0), bootmean1(bootmean1), databoot(databoot),
			alpha(alpha), M_samp(M_samp), B_resamp(B_resamp), w(w), cover(cover) {}   

   // operator
   void operator()(std::size_t begin, std::size_t end) {
   		arma::colvec theta0old= arma::colvec(1);
		arma::colvec theta0new= arma::colvec(1);
		arma::colvec theta1old= arma::colvec(1);
		arma::colvec theta1new= arma::colvec(1); 
		arma::colvec loglikdiff= arma::colvec(1);
		arma::colvec sort0 = arma::colvec(M_samp);
		arma::colvec sort1 = arma::colvec(M_samp);
		arma::colvec r	= arma::colvec(1);r.fill(0.0);
		arma::colvec uu = arma::colvec(1);
		arma::colvec postsamples0	= arma::colvec(M_samp);postsamples0.fill(0.0);
		arma::colvec postsamples1	= arma::colvec(M_samp);postsamples1.fill(0.0);
		double l0;
		double l1;
		double u0;
		double u1;
		for (std::size_t i = begin; i < end; i++) {
			theta0old = thetaboot(i,0);
			theta1old = thetaboot(i,1);
			for(int j=0; j<(M_samp+100); j++) {
				theta0new(0) = R::rnorm(theta0old(0), 0.5);
				loglikdiff(0) = 0.0;
				for(int k=0; k<nn; k++){
					loglikdiff(0) = loglikdiff(0) -w * fabs(databoot(k,2*i+1)-theta0new(0) - theta1old(0)*databoot(k,2*i)) + w * fabs(databoot(k,2*i+1)-theta0old(0) - theta1old(0)*databoot(k,2*i)); 
				}
				r = R::dnorm(theta0new(0), theta0old(0),.5, 0)/R::dnorm(theta0old(0),theta0new(0),.5, 0);
				loglikdiff(0) = loglikdiff(0) + log(r(0));
				loglikdiff(0) = fmin(std::exp(loglikdiff(0)), 1.0);
				uu = R::runif(0.0,1.0);
      				if((uu(0) <= loglikdiff(0)) && (j>99)) {
					postsamples0(j-100) = theta0new(0);
					theta0old(0) = theta0new(0); 
      				}
				else if(j>99){
					postsamples0(j-100) = theta0old(0);	
				}
				theta1new = R::rnorm(theta1old(0), 0.5);
				loglikdiff(0) = 0.0;
				for(int k=0; k<nn; k++){
					loglikdiff(0) = loglikdiff(0) -w * fabs(databoot(k,2*i+1)-theta0old(0) - theta1new(0)*databoot(k,2*i)) + w * fabs(databoot(k,2*i+1)-theta0old(0) - theta1old(0)*databoot(k,2*i)); 
				}
				r = R::dnorm(theta1new(0), theta1old(0),.5, 0) / R::dnorm(theta1old(0),theta1new(0),.5, 0);
				loglikdiff(0) = loglikdiff(0) + log(r(0));
				loglikdiff(0) = fmin(std::exp(loglikdiff(0)), 1.0);
				uu = R::runif(0.0,1.0);
      				if((uu(0) <= loglikdiff(0)) && (j>99)) {
					postsamples1(j-100) = theta1new(0);
					theta1old(0) = theta1new(0); 
      				}
				else if(j>99){
					postsamples1(j-100) = theta1old(0);	
				}
			}
			sort0 = sort(postsamples0);
			sort1 = sort(postsamples1);
			l0 = sort0(0.025*M_samp);
			u0 = sort0(0.975*M_samp);
			l1 = sort1(0.025*M_samp);
			u1 = sort1(0.975*M_samp);
			if ( (l1 < bootmean1(0)) && (u1 > bootmean1(0)) ){
				cover(i) = 1.0;
			} else {cover(i) = 0.0;}			
  		}
	}
};


// [[Rcpp::export]]
arma::colvec rcpp_parallel_qr(SEXP & nn, SEXP & data, SEXP & thetaboot, SEXP & bootmean0, SEXP & bootmean1, SEXP & databoot,
   			SEXP & alpha, SEXP & M_samp, SEXP & B_resamp, SEXP & w) {

   int nn_ = Rcpp::as<int>(nn);
   arma::mat data_ = Rcpp::as<arma::mat>(data);
   arma::mat thetaboot_ = Rcpp::as<arma::mat>(thetaboot); 
   arma::colvec bootmean0_ = Rcpp::as<arma::colvec>(bootmean0);
   arma::colvec bootmean1_ = Rcpp::as<arma::colvec>(bootmean1);
   arma::mat databoot_ = Rcpp::as<arma::mat>(databoot); 
   double alpha_ = Rcpp::as<double>(alpha);
   int M_samp_ = Rcpp::as<int>(M_samp);
   int B_resamp_ = Rcpp::as<int>(B_resamp);
   double w_ = Rcpp::as<double>(w);
	
	
   // allocate the matrix we will return
   arma::colvec cover = arma::colvec(B_resamp_); 

   // create the worker
   GPC_qr_mcmc_parallel gpcWorker(nn_, data_, thetaboot_, bootmean0_, bootmean1_, databoot_, alpha_, M_samp_, B_resamp_, w_, cover);
     
   // call it with parallelFor
   parallelFor(0, B_resamp_, gpcWorker);

   return cover;
}


// [[Rcpp::export]]
Rcpp::List GPC_qr_parallel(SEXP & nn, SEXP & data, SEXP & theta_boot, SEXP & data_boot, SEXP & alpha, SEXP & M_samp, SEXP & B_resamp) {

RNGScope scp;
Rcpp::Function rcpp_parallel_qr("rcpp_parallel_qr");
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
arma::colvec postsamples0f	= arma::colvec(2*M);
arma::colvec postsamples1f	= arma::colvec(2*M);
arma::colvec sort0		= arma::colvec(M);
arma::colvec sort1		= arma::colvec(M);
NumericVector theta0old;
NumericVector theta0new;
NumericVector theta1old;
NumericVector theta1new;
NumericVector loglikdiff;
arma::colvec r			= arma::colvec(1);r.fill(0.0);
arma::colvec uu 		= arma::colvec(1);
arma::colvec cover;
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

// create the worker
cover = Rcpp::as<arma::colvec>(rcpp_parallel_qr(n, ddata, thetaboot, bootmean0, bootmean1, databoot, aalpha, M, B, w));
sumcover = 0.0;
for(int s = 0; s<B; s++){sumcover = sumcover + cover(s);}
diff = (sumcover/B) - (1.0-aalpha);
if(((abs(diff)<= eps)&&(diff>=0)) || t>16) {
   go = FALSE;
} else {
   t = t+1;
   w = fmax(w + (pow(1+t,-0.51)*diff),0.1);
} 
/*
while(go) {
 
   // call it with parallelFor
   parallelFor(0, B, gpcWorker);

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
	double l0;
	double u0;
	double l1;
	double u1;
	l0 = sort0(0.025*2*M);
	u0 = sort0(0.975*2*M);
	l1 = sort1(0.025*2*M);
	u1 = sort1(0.975*2*M);


result = Rcpp::List::create(Rcpp::Named("l0") = l0,Rcpp::Named("u0") = u0,Rcpp::Named("l1") = l1,Rcpp::Named("u1") = u1,Rcpp::Named("w") = w,Rcpp::Named("t") = t,Rcpp::Named("diff") = diff);
*/
result = Rcpp::List::create(Rcpp::Named("w") = w,Rcpp::Named("t") = t,Rcpp::Named("cover") = cover);

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

Rcpp::List GPC_linreg(SEXP & nn, SEXP & data, SEXP & theta_boot, SEXP & data_boot, SEXP & alpha, SEXP & M_samp, SEXP & B_resamp) { 

List result;
double aalpha 			= Rcpp::as<double>(alpha);			 			
int n				= Rcpp::as<int>(nn);
int B 				= Rcpp::as<int>(B_resamp);
double eps 			= 0.01; 
double w			= 0.8;
arma::mat thetaboot     	= Rcpp::as<arma::mat>(theta_boot);
arma::mat ddata			= Rcpp::as<arma::mat>(data);
arma::mat databoot 		= Rcpp::as<arma::mat>(data_boot);
int M				= Rcpp::as<int>(M_samp);
arma::colvec postsamples0	= arma::colvec(M);
arma::colvec postsamples1	= arma::colvec(M);
arma::colvec postsamples2	= arma::colvec(M);
arma::colvec postsamples3	= arma::colvec(M);
arma::colvec postsamples4	= arma::colvec(M);
arma::colvec postsamples0f	= arma::colvec(2*M);
arma::colvec postsamples1f	= arma::colvec(2*M);
arma::colvec postsamples2f	= arma::colvec(2*M);
arma::colvec postsamples3f	= arma::colvec(2*M);
arma::colvec postsamples4f	= arma::colvec(2*M);
arma::colvec sort0		= arma::colvec(M);
arma::colvec sort1		= arma::colvec(M);
arma::colvec sort2		= arma::colvec(M);
arma::colvec sort3		= arma::colvec(M);
double l0;
double l1;
double u0;
double u1;
double l2;
double l3;
double u2;
double u3;
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
	bootmean4 = bootmean4 + thetaboot(i,4);
}
bootmean0 = bootmean0/B;
bootmean1 = bootmean1/B;
bootmean2 = bootmean2/B;
bootmean3 = bootmean3/B;
bootmean4 = bootmean4/B;



while(go) {
for (int i=0; i<B; i++) {
	theta0old = thetaboot(i,0);
	theta1old = thetaboot(i,1);
	theta2old = thetaboot(i,2);
	theta3old = thetaboot(i,3);
	theta4old = thetaboot(i,4);
	for(int j=0; j<(M+100); j++) {
		theta0new = Rcpp::rnorm(1, theta0old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*(1/theta4old)*pow(databoot(k,i)-theta0new*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*(1/theta4old)*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta0new, theta0old[0],.1)/Rcpp::dnorm(theta0old,theta0new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples0(j-100) = theta0new[0];
			theta0old = theta0new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=99)){
			theta0old = theta0new; 
		}else if(j>99){
			postsamples0(j-100) = theta0old[0];	
		}
		theta1new = Rcpp::rnorm(1, theta1old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w *  0.5*(1/theta4old)*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1new*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*(1/theta4old)*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta1new, theta1old[0],.1) / Rcpp::dnorm(theta1old,theta1new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples1(j-100) = theta1new[0];
			theta1old = theta1new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=99)){
			theta1old = theta1new; 
		}else if(j>99){
			postsamples1(j-100) = theta1old[0];	
		}
		theta2new = Rcpp::rnorm(1, theta2old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w *  0.5*(1/theta4old)*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2new*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*(1/theta4old)*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta2new, theta2old[0],.1) / Rcpp::dnorm(theta2old,theta2new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples2(j-100) = theta2new[0];
			theta2old = theta2new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=99)){
			theta2old = theta2new; 
		}else if(j>99){
			postsamples2(j-100) = theta2old[0];	
		}
		theta3new = Rcpp::rnorm(1, theta3old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*(1/theta4old)* pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3new*ddata(k,4),2) + w* 0.5*(1/theta4old)*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta3new, theta3old[0],.1) / Rcpp::dnorm(theta3old,theta3new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples3(j-100) = theta3new[0];
			theta3old = theta3new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=99)){
			theta3old = theta3new; 
		}else if(j>99){
			postsamples3(j-100) = theta3old[0];	
		}
		theta4new = Rcpp::rnorm(1, theta4old[0], 0.1);
		if(theta4new[0]>0){
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -0.5*w*log(theta4new) + 0.5*w*log(theta4old) -w * 0.5*(1/theta4new)* pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*(1/theta4old)*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta4new, theta4old[0],0.1) / Rcpp::dnorm(theta4old,theta4new[0],0.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples4(j-100) = theta4new[0];
			theta4old = theta4new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=99)){
			theta4old = theta4new;	
		}
		else if(j>99){
			postsamples4(j-100) = theta4old[0];	
		}
		}else {
			if(j>99){
			postsamples4(j-100) = theta4old[0];	
		}	
		}
	}
	sort0 = sort(postsamples0);
	sort1 = sort(postsamples1);
	sort2 = sort(postsamples2);
	sort3 = sort(postsamples3);
	l0 = sort0(0.025*M);
	u0 = sort0(0.975*M);
	l1 = sort1(0.025*M);
	u1 = sort1(0.975*M);
	l2 = sort2(0.025*M);
	u2 = sort2(0.975*M);
	l3 = sort3(0.025*M);
	u3 = sort3(0.975*M);
	if ( (l3 < bootmean3(0)) && (u3 > bootmean3(0)) && (l2 < bootmean2(0)) && (u2 > bootmean2(0)) && (l1 < bootmean1(0)) && (u1 > bootmean1(0))){
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
	w = fmax(w + (pow(1+t,-0.51)*2*diff),0.6);
}
}

theta0old[0] = bootmean0(0);
theta1old[0] = bootmean1(0);
theta2old[0] = bootmean2(0);
theta3old[0] = bootmean3(0);
theta4old[0] = bootmean4(0);
	for(int j=0; j<(2*M+1000); j++) {
		theta0new = Rcpp::rnorm(1, theta0old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*(1/theta4old)*  pow(ddata(k,0)-theta0new*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*(1/theta4old)* pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta0new, theta0old[0],.1)/Rcpp::dnorm(theta0old,theta0new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples0f(j-1000) = theta0new[0];
			theta0old = theta0new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=999)){
			theta0old = theta0new; 
		}
		else if(j>999){
			postsamples0f(j-1000) = theta0old[0];	
		}
		theta1new = Rcpp::rnorm(1, theta1old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*(1/theta4old)*  pow(ddata(k,0)-theta0old*ddata(k,1)-theta1new*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*(1/theta4old)* pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta1new, theta1old[0],.1) / Rcpp::dnorm(theta1old,theta1new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples1f(j-1000) = theta1new[0];
			theta1old = theta1new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=999)){
			theta1old = theta1new; 
		}
		else if(j>999){
			postsamples1f(j-1000) = theta1old[0];	
		}
		theta2new = Rcpp::rnorm(1, theta2old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*(1/theta4old)*  pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2new*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*(1/theta4old)* pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta2new, theta2old[0],.1) / Rcpp::dnorm(theta2old,theta2new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples2f(j-1000) = theta2new[0];
			theta2old = theta2new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=999)){
			theta2old = theta2new; 
		}
		else if(j>999){
			postsamples2f(j-1000) = theta2old[0];	
		}
		theta3new = Rcpp::rnorm(1, theta3old[0], 0.1);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*(1/theta4old)*  pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3new*ddata(k,4),2) + w* 0.5*(1/theta4old)* pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta3new, theta3old[0],.1) / Rcpp::dnorm(theta3old,theta3new[0],.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples3f(j-1000) = theta3new[0];
			theta3old = theta3new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=999)){
			theta3old = theta3new; 
		}
		else if(j>999){
			postsamples3f(j-1000) = theta3old[0];	
		}
		theta4new = Rcpp::rnorm(1, theta4old[0], 0.1);
		if(theta4new[0]>0){
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -0.5*w*log(theta4new) + 0.5*w*log(theta4old) -w * 0.5*(1/theta4new)* pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*(1/theta4old)*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta4new, theta4old[0],0.1) / Rcpp::dnorm(theta4old,theta4new[0],0.1);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples4f(j-1000) = theta4new[0];
			theta4old = theta4new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=999)){
			theta4old = theta4new; 
		}
		else if(j>999){
			postsamples4f(j-1000) = theta4old[0];	
		}
		}else {
		if(j>999){
			postsamples4f(j-1000) = theta4old[0];	
		}	
		}
	}
	sort0 = sort(postsamples0f);
	sort1 = sort(postsamples1f);
	sort2 = sort(postsamples2f);
	sort3 = sort(postsamples3f);
	double fvar = 0.0;
	for(int k=0; k<2*M; k++){
		fvar = fvar+(postsamples4f(k)/(2*M)); 
	}
	l0 = sort0(0.025*2*M);
	u0 = sort0(0.975*2*M);
	l1 = sort1(0.025*2*M);
	u1 = sort1(0.975*2*M);
	l2 = sort2(0.025*2*M);
	u2 = sort2(0.975*2*M);
	l3 = sort3(0.025*2*M);
	u3 = sort3(0.975*2*M);
	intvs9080(0) = sort0(0.05*2*M);
	intvs9080(1) = sort0(0.95*2*M);
	intvs9080(2) = sort1(0.05*2*M);
	intvs9080(3) = sort1(0.95*2*M);
	intvs9080(4) = sort2(0.05*2*M);
	intvs9080(5) = sort2(0.95*2*M);
	intvs9080(6) = sort3(0.05*2*M);
	intvs9080(7) = sort3(0.95*2*M);
	intvs9080(8) = sort0(0.10*2*M);
	intvs9080(9) = sort0(0.90*2*M);
	intvs9080(10) = sort1(0.10*2*M);
	intvs9080(11) = sort1(0.90*2*M);
	intvs9080(12) = sort2(0.10*2*M);
	intvs9080(13) = sort2(0.90*2*M);
	intvs9080(14) = sort3(0.10*2*M);
	intvs9080(15) = sort3(0.90*2*M);


result = Rcpp::List::create(Rcpp::Named("l0") = l0,Rcpp::Named("u0") = u0,Rcpp::Named("l1") = l1,Rcpp::Named("u1") = u1,Rcpp::Named("l2") = l2,Rcpp::Named("u2") = u2,Rcpp::Named("l3") = l3,Rcpp::Named("u3") = u3,Rcpp::Named("w") = w,Rcpp::Named("diff") = diff,Rcpp::Named("t") = t,Rcpp::Named("intvs9080") = intvs9080,Rcpp::Named("postvar")=fvar );
return result;
}
/*
// [[Rcpp::export]]

Rcpp::List GPC_linreg(SEXP & nn, SEXP & data, SEXP & theta_boot, SEXP & data_boot, SEXP & alpha, SEXP & M_samp, SEXP & B_resamp) { 

List result;
double aalpha 			= Rcpp::as<double>(alpha);			 			
int n				= Rcpp::as<int>(nn);
int B 				= Rcpp::as<int>(B_resamp);
double eps 			= 0.01; 
double w			= 0.3;
arma::mat thetaboot     	= Rcpp::as<arma::mat>(theta_boot);
arma::mat ddata			= Rcpp::as<arma::mat>(data);
arma::mat databoot 		= Rcpp::as<arma::mat>(data_boot);
int M				= Rcpp::as<int>(M_samp);
arma::colvec postsamples0	= arma::colvec(M);
arma::colvec postsamples1	= arma::colvec(M);
arma::colvec postsamples2	= arma::colvec(M);
arma::colvec postsamples3	= arma::colvec(M);
arma::colvec postsamples0f	= arma::colvec(2*M);
arma::colvec postsamples1f	= arma::colvec(2*M);
arma::colvec postsamples2f	= arma::colvec(2*M);
arma::colvec postsamples3f	= arma::colvec(2*M);
arma::colvec sort0		= arma::colvec(M);
arma::colvec sort1		= arma::colvec(M);
arma::colvec sort2		= arma::colvec(M);
arma::colvec sort3		= arma::colvec(M);
double l0;
double l1;
double u0;
double u1;
double l2;
double l3;
double u2;
double u3;
arma::colvec intvs9080	= arma::colvec(16);
NumericVector theta0old;
NumericVector theta0new;
NumericVector theta1old;
NumericVector theta1new;
NumericVector theta2old;
NumericVector theta2new;
NumericVector theta3old;
NumericVector theta3new;	
NumericVector loglikdiff;
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



while(go) {
for (int i=0; i<B; i++) {
	theta0old = thetaboot(i,0);
	theta1old = thetaboot(i,1);
	theta2old = thetaboot(i,2);
	theta3old = thetaboot(i,3);
	for(int j=0; j<(M+100); j++) {
		theta0new = Rcpp::rnorm(1, theta0old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*pow(databoot(k,i)-theta0new*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta0new, theta0old[0],.5)/Rcpp::dnorm(theta0old,theta0new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples0(j-100) = theta0new[0];
			theta0old = theta0new; 
      		}
		else if((uu(0) <= loglikdiff[0]) && (j<=99)) {
			theta0old = theta0new; 
      		} 
		else if(j>99){
			postsamples0(j-100) = theta0old[0];	
		}
		theta1new = Rcpp::rnorm(1, theta1old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w *  0.5*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1new*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta1new, theta1old[0],.5) / Rcpp::dnorm(theta1old,theta1new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples1(j-100) = theta1new[0];
			theta1old = theta1new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=99)) {
			theta1old = theta1new; 
      		} 
		else if(j>99){
			postsamples1(j-100) = theta1old[0];	
		}
		theta2new = Rcpp::rnorm(1, theta2old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w *  0.5*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2new*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta2new, theta2old[0],.5) / Rcpp::dnorm(theta2old,theta2new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples2(j-100) = theta2new[0];
			theta2old = theta2new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=99)) {
			theta2old = theta2new; 
      		} 
		else if(j>99){
			postsamples2(j-100) = theta2old[0];	
		}
		theta3new = Rcpp::rnorm(1, theta3old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5* pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3new*ddata(k,4),2) + w* 0.5*pow(databoot(k,i)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta3new, theta3old[0],.5) / Rcpp::dnorm(theta3old,theta3new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>99)) {
			postsamples3(j-100) = theta3new[0];
			theta3old = theta3new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=99)) {
			theta3old = theta3new; 
      		} 
		else if(j>99){
			postsamples3(j-100) = theta3old[0];	
		}
	}
	sort0 = sort(postsamples0);
	sort1 = sort(postsamples1);
	sort2 = sort(postsamples2);
	sort3 = sort(postsamples3);
	l0 = sort0(0.025*M);
	u0 = sort0(0.975*M);
	l1 = sort1(0.025*M);
	u1 = sort1(0.975*M);
	l2 = sort2(0.025*M);
	u2 = sort2(0.975*M);
	l3 = sort3(0.025*M);
	u3 = sort3(0.975*M);
	if ( (l3 < bootmean3(0)) && (u3 > bootmean3(0)) && (l2 < bootmean2(0)) && (u2 > bootmean2(0)) && (l1 < bootmean1(0)) && (u1 > bootmean1(0))){
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
	w = fmax(w + (pow(1+t,-0.51)*2*diff),0.01);
}
}

theta0old[0] = bootmean0(0);
theta1old[0] = bootmean1(0);
theta2old[0] = bootmean2(0);
theta3old[0] = bootmean3(0);
	for(int j=0; j<(2*M+1000); j++) {
		theta0new = Rcpp::rnorm(1, theta0old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*pow(ddata(k,0)-theta0new*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta0new, theta0old[0],.5)/Rcpp::dnorm(theta0old,theta0new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples0f(j-1000) = theta0new[0];
			theta0old = theta0new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=999)) {
			theta0old = theta0new; 
      		}
		else if(j>999){
			postsamples0f(j-1000) = theta0old[0];	
		}
		theta1new = Rcpp::rnorm(1, theta1old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1new*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta1new, theta1old[0],.5) / Rcpp::dnorm(theta1old,theta1new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples1f(j-1000) = theta1new[0];
			theta1old = theta1new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=999)) {
			theta1old = theta1new; 
      		}
		else if(j>999){
			postsamples1f(j-1000) = theta1old[0];	
		}
		theta2new = Rcpp::rnorm(1, theta2old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2new*ddata(k,3)-theta3old*ddata(k,4),2) + w* 0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta2new, theta2old[0],.5) / Rcpp::dnorm(theta2old,theta2new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples2f(j-1000) = theta2new[0];
			theta2old = theta2new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=999)) {
			theta2old = theta2new; 
      		}
		else if(j>999){
			postsamples2f(j-1000) = theta2old[0];	
		}
		theta3new = Rcpp::rnorm(1, theta3old[0], 0.5);
		loglikdiff = 0.0;
		for(int k=0; k<n; k++){
			loglikdiff = loglikdiff -w * 0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3new*ddata(k,4),2) + w* 0.5*pow(ddata(k,0)-theta0old*ddata(k,1)-theta1old*ddata(k,2)-theta2old*ddata(k,3)-theta3old*ddata(k,4),2); 
		}
		r = Rcpp::dnorm(theta3new, theta3old[0],.5) / Rcpp::dnorm(theta3old,theta3new[0],.5);
		loglikdiff[0] = loglikdiff[0] + log(r(0));
		loglikdiff[0] = fmin(std::exp(loglikdiff[0]), 1.0);
		uu = Rcpp::runif(1);
      		if((uu(0) <= loglikdiff[0]) && (j>999)) {
			postsamples3f(j-1000) = theta3new[0];
			theta3old = theta3new; 
      		}else if((uu(0) <= loglikdiff[0]) && (j<=999)) {
			theta3old = theta3new; 
      		}
		else if(j>999){
			postsamples3f(j-1000) = theta3old[0];	
		}
	}
	sort0 = sort(postsamples0f);
	sort1 = sort(postsamples1f);
	sort2 = sort(postsamples2f);
	sort3 = sort(postsamples3f);
	l0 = sort0(0.025*2*M);
	u0 = sort0(0.975*2*M);
	l1 = sort1(0.025*2*M);
	u1 = sort1(0.975*2*M);
	l2 = sort2(0.025*2*M);
	u2 = sort2(0.975*2*M);
	l3 = sort3(0.025*2*M);
	u3 = sort3(0.975*2*M);
	intvs9080(0) = sort0(0.05*2*M);
	intvs9080(1) = sort0(0.95*2*M);
	intvs9080(2) = sort1(0.05*2*M);
	intvs9080(3) = sort1(0.95*2*M);
	intvs9080(4) = sort2(0.05*2*M);
	intvs9080(5) = sort2(0.95*2*M);
	intvs9080(6) = sort3(0.05*2*M);
	intvs9080(7) = sort3(0.95*2*M);
	intvs9080(8) = sort0(0.10*2*M);
	intvs9080(9) = sort0(0.90*2*M);
	intvs9080(10) = sort1(0.10*2*M);
	intvs9080(11) = sort1(0.90*2*M);
	intvs9080(12) = sort2(0.10*2*M);
	intvs9080(13) = sort2(0.90*2*M);
	intvs9080(14) = sort3(0.10*2*M);
	intvs9080(15) = sort3(0.90*2*M);


result = Rcpp::List::create(Rcpp::Named("l0") = l0,Rcpp::Named("u0") = u0,Rcpp::Named("l1") = l1,Rcpp::Named("u1") = u1,Rcpp::Named("l2") = l2,Rcpp::Named("u2") = u2,Rcpp::Named("l3") = l3,Rcpp::Named("u3") = u3,Rcpp::Named("w") = w,Rcpp::Named("diff") = diff,Rcpp::Named("t") = t,Rcpp::Named("intvs9080") = intvs9080);
return result;
}
*/

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
