# Optimal-congestion-management-in-power-systems

This file contains the work I have done for my MSc dissertation project.


The data for MILP model is named as MILP_data.xlsx file. The main model for Mixed Integer Programming for congestion management using load shedding, line switching and phase angle regulators. In this model, binary variables for congestion management method indicators (w1, w2, w3) can be used as parameters or decision variables depending on which results are required to the user.



If the user requires only the optimal soultion then using w1, w2, w3 as decision varibales is suggested. If the user wants to compare the results obtained from different combinations of the congestion management techniques, then using w1, w2, w3 as parameters is recommended.


The power generation by the economic dispatch is done by using the code economic_dispatch.py and data from ED_data.xlsx file.



The results for Load sensitivity matrix, PTDF matrix, power flow through transmission lines and voltage angles for each bus are obtained by using the code ptdf.py and data for PTDF method is named as PTDF_data.xlsx file.


