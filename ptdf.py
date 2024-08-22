import pandas as pd
import numpy as np

def read_and_prepare_data(file_path):
    # Read Excel data
    reactances = pd.read_excel(file_path, sheet_name='Reactances', index_col=0)
    adjacency_matrix = pd.read_excel(file_path, sheet_name='AdjacencyMatrix', index_col=0)
    bus_data = pd.read_excel(file_path, sheet_name='BusData', index_col=0)

    # Prepare reactance matrix
    X_diag = reactances['Reactance'].values

    # Prepare adjacency matrix, remove reference bus row (Bus13)
    A = adjacency_matrix.drop('Bus13', axis=0).values
    bus_labels = adjacency_matrix.index.drop('Bus13')  # Labels for buses without reference bus

    # Calculate pB, net flow out of network (load - generation)
    pB = bus_data['Load'] - bus_data['Generation']
    pB = pB.drop('Bus13')  # Remove reference bus from pB

    return X_diag, A, pB.values, bus_labels, reactances.index


def compute_matrices_and_vectors(X_diag, A, pB, bus_labels, line_labels):
    # Convert reactance array to diagonal matrix
    X = np.diag(X_diag)

    # Compute inverse of X
    X_inv = np.linalg.inv(X)

    # Compute AX^(-1)A^T and its inverse
    AX_inv_AT = A @ X_inv @ A.T
    FA = -np.linalg.inv(AX_inv_AT)
    FP = X_inv @ A.T @ np.linalg.inv(AX_inv_AT)

    # Compute delta and pL using equation (1) and (2) from the document
    delta = FA @ pB
    pL = FP @ pB

    # Creating DataFrames for Excel export
    FA_df = pd.DataFrame(FA, index=bus_labels, columns=bus_labels)
    FP_df = pd.DataFrame(FP, index=line_labels, columns=bus_labels)
    pL_df = pd.DataFrame(pL, index=line_labels, columns=['p^L'])
    delta_df = pd.DataFrame(delta, index=bus_labels, columns=['delta_B'])

    return FA_df, FP_df, pL_df, delta_df


def export_to_excel(FA_df, FP_df, pL_df, delta_df, output_file):
    with pd.ExcelWriter(output_file) as writer:
        FA_df.to_excel(writer, sheet_name='F^A')
        FP_df.to_excel(writer, sheet_name='F^P')
        pL_df.to_excel(writer, sheet_name='p^L')
        delta_df.to_excel(writer, sheet_name='delta')


# File paths
file_path = PTDF_data.xlsx
output_file = 'output_matrices.xlsx'

# Read and prepare data
X_diag, A, pB, bus_labels, line_labels = read_and_prepare_data(file_path)

# Compute matrices and vectors
FA_df, FP_df, pL_df, delta_df = compute_matrices_and_vectors(X_diag, A, pB, bus_labels, line_labels)

# Export to Excel
export_to_excel(FA_df, FP_df, pL_df, delta_df,output_file)

print("Matrices and vector exported to Excel successfully.")
