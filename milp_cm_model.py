import pandas as pd
import numpy as np
import docplex.mp.model as cpx
import cplex

def load_data(file_path):
    # Load sets
    df_sets = pd.read_excel(file_path, sheet_name='Sets')
    B = df_sets['B'].dropna().astype(int).tolist()
    L = df_sets['L'].dropna().astype(int).tolist()
    L_o = df_sets['L_o'].dropna().astype(int).tolist()
    L_s = df_sets['L_s'].dropna().astype(int).tolist()
    L_p = df_sets['L_p'].dropna().astype(int).tolist()
    W = df_sets['W'].dropna().astype(int).tolist()

    # Load parameters
    df_params = pd.read_excel(file_path, sheet_name='Parameters')
    x_l = df_params['x_l'].dropna().to_numpy()
    PL_UB = df_params['PL_UB'].dropna().to_numpy()
    PG_UB = df_params['PG_UB'].dropna().to_numpy()
    PL_LB = df_params['PL_LB'].dropna().to_numpy()
    PG_LB = df_params['PG_LB'].dropna().to_numpy()
    Cg = df_params['Cg'].dropna().to_numpy()

    # Load matrices
    A = pd.read_excel(file_path, sheet_name='A', header=0, index_col=0).to_numpy()
    PDt = pd.read_excel(file_path, sheet_name='PDt', header=0, index_col=0).to_numpy()
    Cb = pd.read_excel(file_path, sheet_name='Cb', header=0, index_col=0).to_numpy()
    Cl = pd.read_excel(file_path, sheet_name='Cl', header=0, index_col=0).to_numpy()
    Cp = pd.read_excel(file_path, sheet_name='Cp', header=0, index_col=0).to_numpy()

    return B, L, L_o, L_s, L_p, W, x_l, A, PL_UB, PG_UB, PL_LB, PG_LB, Cg, PDt, Cb, Cl, Cp

def solve_model(file_path):
    B, L, L_o, L_s, L_p, W, x_l, A, PL_UB, PG_UB, PL_LB, PG_LB, Cg, PDt, Cb, Cl, Cp = load_data(file_path)

    T = [1]  # Time periods
    Ht = [0, 1]  # Length of time periods
    G = B
    M = 10000  # Big M for line switching constraints

    ### Model
    problem = cpx.Model(name="Congestion Management")

    ### Variables
    # Power output of generator g
    pGgt = {(g, t): problem.continuous_var(lb=PG_LB[g], ub=PG_UB[g], name="pGgt_{0}_{1}".format(g, t)) for g in G for t
            in T}
    print(*list(pGgt.values()), sep="\n")

    # Power flow in line l
    pLlt = {(l, t): problem.continuous_var(lb=PL_LB[l], ub=PL_UB[l], name="pLlt_{0}_{1}".format(l, t)) for l in L for t
            in T}
    print(*list(pLlt.values()), sep="\n")

    # Voltage phase angle at bus b
    delta_bt = {(b, t): problem.continuous_var(lb=-np.inf, ub=np.inf, name="delta_bt_{0}_{1}".format(b, t)) for b in B
                for t in T}


    # Voltage phase angle at bus 13 should be zero (reference bus constraint)
    reference_bus_constraints = {t: problem.add_constraint(
        ct=delta_bt[13, t] == 0,
        ctname="reference_bus_cons_t_{0}".format(t)) for t in T}
    print(*list(reference_bus_constraints.values()), sep="\n")


    # proportion of the load supplied at bus b
    k_bt = {(b, t): problem.continuous_var(lb=0, ub=1, name="k_bt_{0}_{1}".format(b, t)) for b in B for t in T}

    # Variable for line l to switch on or off
    y_lt = {(l, t): problem.binary_var(name="y_lt_{0}_{1}".format(l, t)) for l in L_s for t in T}

    # PAR maximum limit
    p_limit = 0.5
    # Phase shift introduced by the PAR on line
    PAR = problem.continuous_var(lb=-p_limit, ub=p_limit, name="PAR")
    # PAR tightening variable
    PAR2 = problem.continuous_var(lb=-np.inf, ub=np.inf, name="PAR2")

    # Model combination Variable
    w = np.array([0,1,1,1]) # w1, w2, w3  # if used as parameters
    #w = {ww: problem.binary_var(name="w_{0}".format(ww)) for ww in W}  # w1, w2, w3  # if used as variable

    ### Constraints
    # KCL constraint
    KCL_cons = {(b, t): problem.add_constraint(
        ct= pGgt[b, t] + problem.sum(A[b, l] * pLlt[l, t] for l in L) == k_bt[b, t] * PDt[t, b],
        ctname="KCL_cons_{0}_{1}".format(b, t)) for t in T for b in B}
    print(*list(KCL_cons.values()), sep="\n")

    # KVL for ordinary lines
    KVL_con_standard = {(l, t): problem.add_constraint(
        ct= x_l[l]*pLlt[l, t] + problem.sum(A[b, l] * delta_bt[b, t] for b in B) == 0,
        ctname="KVL_con_standard_{0}_{1}".format(l, t)) for t in T for l in L_o}
    print(*list(KVL_con_standard.values()), sep="\n")

    # KVL for lines with switching capability
    KVL_con_line_switching_1 = {(l, t): problem.add_constraint(
        ct=-(1 - y_lt[l, t]) * M <= x_l[l]*pLlt[l, t] + problem.sum(A[b, l] * delta_bt[b, t] for b in B),
        ctname="KVL_con_line_switching1_{0}_{1}".format(l, t)) for t in T for l in L_s}
    print(*list(KVL_con_line_switching_1.values()), sep="\n")

    KVL_con_line_switching_2 = {(l, t): problem.add_constraint(
        ct= x_l[l]*pLlt[l, t] + problem.sum(A[b, l] * delta_bt[b, t] for b in B) <= (1 - y_lt[l, t]) * M,
        ctname="KVL_con_line_switching2_{0}_{1}".format(l, t)) for t in T for l in L_s}
    print(*list(KVL_con_line_switching_2.values()), sep="\n")

    # KVL for lines with PAR
    KVL_con_PAR = {(l, t): problem.add_constraint(
        ct= x_l[l]*pLlt[l, t] + (problem.sum(A[b, l] * delta_bt[b, t] for b in B) + (PAR)) == 0,
        ctname="KVL_con_PAR_{0}_{1}".format(l, t)) for t in T for l in L_p}
    print(*list(KVL_con_PAR.values()), sep="\n")

    # Line Flow Limit for lines with switching capability
    PLT_con_line_switching_1 = {(l, t): problem.add_constraint(
        ct=y_lt[l, t] * PL_LB[l] <= pLlt[l, t],
        ctname="PLT_con_line_switching1_{0}_{1}".format(l, t)) for t in T for l in L_s}
    print(*list(PLT_con_line_switching_1.values()), sep="\n")

    PLT_con_line_switching_2 = {(l, t): problem.add_constraint(
        ct=pLlt[l, t] <= y_lt[l, t] * PL_UB[l],
        ctname="PLT_con_line_switching2_{0}_{1}".format(l, t)) for t in T for l in L_s}
    print(*list(PLT_con_line_switching_2.values()), sep="\n")

    # Line Switching constraint
    yLT_con_line_switching = {(l, t): problem.add_constraint(
        ct=1 - w[2] <= y_lt[l, t],
        ctname="yLT_line_switching_{0}_{1}".format(l, t)) for t in T for l in L_s}
    print(*list(yLT_con_line_switching.values()), sep="\n")

    # PAR binding constraints
    PAR_1 = {(l): problem.add_constraint(ct=w[3] * (-p_limit) <= PAR, ctname="PAR_con_{0}".format(l)) for l in L_p}
    print(*list(PAR_1.values()), sep="\n")
    PAR_2 = {(l): problem.add_constraint(ct=PAR <= w[3] * p_limit, ctname="PAR_con_{0}".format(l)) for l in L_p}
    print(*list(PAR_2.values()), sep="\n")

    # Load shedding constraint
    load_shed_con = {(b, t): problem.add_constraint(
        ct=1 - w[1] <= k_bt[b, t],
        ctname="load_shed_con_{0}_{1}".format(b, t)) for t in T for b in B}
    print(*list(load_shed_con.values()), sep="\n")

    # Constraints to make PAR absolute
    p_con1 = problem.add_constraint(ct=PAR <= PAR2, ctname="p_con1")
    print("%s" % str(p_con1))
    p_con2 = problem.add_constraint(ct=PAR >= -PAR2, ctname="p_con2")
    print("%s" % str(p_con2))

    # Model combination constraint
    w_con = problem.add_constraint(ct=(problem.sum(w[ww] for ww in W)) <= 3, ctname="w_con")
    print("%s" % str(w_con))


    ### Objective function
    objective = problem.sum(Ht[t] * pGgt[g, t] * Cg[g] for t in T for g in G) + \
                w[1] * problem.sum(Cb[t, b] * (1 - k_bt[b, t]) * PDt[t, b] for t in T for b in B) + \
                w[2] * problem.sum(Cl[t, l] * (1 - y_lt[l, t]) for l in L_s for t in T) + \
                w[3] * problem.sum(Cp[t, l] * PAR2 for l in L_p for t in T)
    print(objective)


    problem.minimize(objective)
    problem.print_information()
    sol = problem.solve(log_output=True)
    problem.print_solution()
    print("Objective Value: ", sol.get_objective_value())
    print("Solve Time: ",sol.solve_details.time)

def main(file_path):
    solve_model(file_path)

file_path = "C:/Users/s2511090/OneDrive - University of Edinburgh/Desktop/dissertation_data.xlsx"

if __name__ == "__main__":
    main(file_path)
