\name{GPC_linreg}
\alias{GPC_linreg}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Calibration algorithm for linear regression example 4.2
}
\description{
 
}
\usage{
GPC_linreg(nn, data, theta_boot, data_boot, alpha, M_samp, B_resamp)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{nn}{
	Number of data points.
}
  \item{data}{
	Data.
}
  \item{theta_boot}{
	B bootstrap estimates of the parameter.
}
  \item{data_boot}{
	B bootstrap resampled data sets.
}
  \item{alpha}{
	Nominal coverage probability is given by 100*(1-alpha)%.
}
  \item{M_samp}{
	Number of posterior samples.
}
  \item{B_resamp}{
	Number of bootstrap resamples.
}
}
\value{
 
}

\examples{
\dontrun{
 
}
}

