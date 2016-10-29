# MPC algorithm


MPC R Codes includes an R script to implement the quantile regression example in https://arxiv.org/abs/1509.00922.

Additionally, the MPC folder contains files needed to install an MPC package with Rcpp codes for all three examples done in the above paper.

To install the MPC package you will need:
  1.  R, version 3.3.q was used to build MPC.
  2.  Rtools.
  3.  The devtools package.
  
Steps:
  1.  Open Rgui.
  2.  Install the "devtools" package if you have not already using the dropdown "Packages -> Install package(s)...".  Load the devtools      package using the command library(devtools).  
  3.  Submit the command: install_github("nasyring/MPC", subdir = "MPC").
