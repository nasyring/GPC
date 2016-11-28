\name{GPC_varmix}
\alias{GPC_varmix}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
GPC for variational posterior of a normal mixture model
}
\description{
calibrate the variational posterior for a normal mixture model
}
\usage{
GPC_varmix(nn, data, data_boot, alpha, B_resamp)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{nn}{
	Number of data points.
}
  \item{data}{
	Data.
}
  \item{data_boot}{
	B bootstrap resampled data sets.
}
  \item{alpha}{
	Nominal coverage probability is given by 100*(1-alpha)%.
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

