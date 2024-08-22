import pandas as pd
from docplex.mp.model import Model

# Load data from Excel
file_path = D_Data.xlsx
df_generators = pd.read_excel(file_path, sheet_name='Generators')
df_buses = pd.read_excel(file_path, sheet_name='Buses')
df_lines = pd.read_excel(file_path, sheet_name='Lines')

# Initialize the model
mdl = Model('EconomicDispatch')

# Variables
# Generator outputs
df_generators['output'] = mdl.continuous_var_list(len(df_generators), lb=df_generators['P_min'], ub=df_generators['P_max'], name='Output')

# Objective: Minimize total generation cost
total_cost = mdl.sum(df_generators['C_g'] * df_generators['output'])

# Power balance
mdl.add_constraint(mdl.sum(df_generators['output']) == mdl.sum(df_buses['P_D']))

mdl.minimize(total_cost)
# Solve the problem
solution = mdl.solve(log_output=True)

# Print the solution
if solution:
    print("Solution status: ", mdl.get_solve_status())
    print("Minimum cost: ", mdl.objective_value)
    for gen in df_generators.itertuples():
        print(f"Generator {gen.Index} at Bus {gen.bus_id}: Output = {gen.output.solution_value}")
else:
    print("No solution found")
