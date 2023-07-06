Replication files for Keweloh, Sascha A., Stephan Hetzenecker, and Andre Seepe. "Monetary Policy and Information Shocks in a Block-Recursive SVAR", forthcoming in the Journal of International Money and Finance




The data files are  available on request.

1. The variables (data\MPandStock\dataJK.csv) used in the monthly SVAR are downloaded from 
https://www.aeaweb.org/articles?id=10.1257/mac.20180090
Please refer to: Jarociński, Marek, and Peter Karadi. "Deconstructing monetary policy surprises—the role of information shocks." American Economic Journal: Macroeconomics 12.2 (2020): 1-43.

2. The variables (data\MPandStock\hfdata.csv) used in the high-frequency SVAR and the updated variables (data\MPandStock\hfdataUpdate.csv) are downloaded from 
https://onlinelibrary.wiley.com/doi/10.1111/jofi.13163
Please refer to: Gürkaynak, Refet, Hati̇ce Gökçe Karasoy‐Can, and Sang Seok Lee. "Stock market's assessment of monetary policy transmission: The cash flow effect." The Journal of Finance 77.4 (2022): 2375-2421.
The variables ff4_hf, sp500_hf, and ONRUN2 in "data\MPandStock\hfdata.csv" correspond to the variables  FF4, SP500, and ONRUN2 in "Replication Code-20190750\GSSrawdata.xlsx".



*** Code - Monte Carlo Simulation ***

Generated with Python 3.8.16


- MC_MPandStock_1.py: This file contains the code for the first MC in Section 3.2 and Appendix B. The results are saved in the folder "MCResults".

- MC_MPandStock_2.py: This file contains the code for the MC with the proxy variable in Section 3.2. The results are saved in the folder "MCResults".

- MC_MPandStock_3.py: This file contains the code for the MC for the VAR reported in Appendix B. The results are saved in the folder "MCResults".

- aMC_Evaluation_MPandStock.py and aaMC_Evaluation_MPandStock_irf.py: Use the results saved in the folder "MCResults" to generate the tables and figures in Section 3.2 and Appendix B. All results are saved in the folder "FiguresAndTables".



*** Code - Empirical Analysis ***

- Application_MPandStock_HF.py: Contains the high-frequency SVAR in Section 4.2. The results are saved in the folder "FiguresAndTables".

- Application_MPandStock_HF_update.py: Contains the high-frequency SVAR in Section 4.2 with the updated variables until 2019. The results are saved in the folder "FiguresAndTables".

- Application_MPandStock.py: Contains the monthly SVAR in Section 4.1. The results are saved in the folder "FiguresAndTables".

- Application_MPandStockRob2.py, Application_MPandStockRob3.py, Application_MPandStockRob4.py: Contains the robustness checks in Appendix C. The results are saved in the folder "FiguresAndTables".


*** Proxy variables ***

- The high-frequency shocks and proxy variables from the high-frequency SVAR are saved in the folder "FiguresAndTables" with the name "hf_shocks.csv" and "hf_proxy_variables.csv". 