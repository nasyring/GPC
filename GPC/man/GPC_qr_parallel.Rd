\name{GPC_qr_parallel}
\alias{GPC_qr_parallel}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
GPC for quantile regression example 4.1.
}
\description{
scale the posterior for quantile reg
}
\usage{
GPC_qr_parallel(nn, data, theta_boot, data_boot, alpha, M_samp, B_resamp)
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

