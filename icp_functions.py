import numpy as np
import time
import plotly.graph_objects as go

def Data_Generation():
    # Paramètres de la grille
    m = 80  # Largeur de la grille
    n = m**2  # Nombre de points

    # Création de la grille X, Y et calcul de Z
    x = np.linspace(-2, 2, m)
    y = np.linspace(-2, 2, m)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = np.sin(X) * np.cos(Y)

    # Création de la matrice de points D
    D = np.array([X, Y, Z])

    # Valeurs de translation
    Tx, Ty, Tz = 0.5, -0.3, 0.2
    T = np.array([Tx, Ty, Tz]).reshape(3, 1)

    # Valeurs de rotation (en radians)
    rx, ry, rz = 0.3, -0.2, 0.05
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    R = Rx @ Ry @ Rz

    # Transformation des données
    M = R @ D + T

    # Point cloud data P
    P = D.T

    # Point cloud data Q
    Q = M.T
    return P, Q


def icp(P, Q):
    """
    Aligns the point cloud P to the reference point cloud Q using the Iterative Closest Point (ICP) algorithm.

    Args:
        P (np.array): Moving point cloud, shape (n, 3).
        Q (np.array): Reference point cloud, shape (m, 3).

    Returns:
        R_opt (np.array): Optimal rotation matrix.
        T_opt (np.array): Optimal translation vector.
        E_r (list): Error at each iteration.
        t_c (list): Time taken for each iteration.
    """

    # Initialize variables
    difference = np.zeros_like(Q)  # Stores differences between points
    mapPoint = np.zeros_like(P)   # Stores the closest points in Q for each point in P
    distance = np.zeros(Q.shape[0])  # Distance array for each point in Q
    n = P.shape[0]  # Number of points in P

    j = 0  # Iteration counter
    d = 100  # Initial error value, set high to start the loop

    E_r = []  # To store error at each iteration
    t_c = []  # To store computation time for each iteration

    start_time = time.time()  # Start the overall timer

    # ICP Iteration
    while d > 0.001 and j < 100:  # Stop when error is below threshold or max iterations reached
        j += 1
        iter_start_time = time.time()  # Timer for this iteration

        # Find the closest points in Q for each point in P
        for i in range(n):
            difference = Q - P[i, :]  # Compute difference between P[i] and all points in Q
            difference = difference ** 2  # Square the differences
            distance = np.sum(difference, axis=1)  # Sum differences to compute squared Euclidean distances
            minIndex = np.argmin(distance)  # Index of the closest point in Q
            mapPoint[i, :] = Q[minIndex, :]  # Store the closest point

        # Compute centroids of P and mapped points
        centerP = np.mean(P, axis=0)  # Centroid of P
        centerMap = np.mean(mapPoint, axis=0)  # Centroid of mapped points

        # Center the points around their centroids
        tempP = P - centerP
        tempMapPoint = mapPoint - centerMap

        # Compute the cross-covariance matrix
        H = np.dot(tempP.T, tempMapPoint)

        # Singular Value Decomposition (SVD) to compute rotation
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)  # Rotation matrix

        # Ensure that the rotation matrix is a proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Compute the translation vector
        T = -np.dot(R, centerP) + centerMap

        # Update P by applying the computed transformation (R and T)
        P = (np.dot(R, P.T).T + T)

        # Compute mean squared error for stopping condition
        d = np.sum(np.sum((P - mapPoint) ** 2, axis=1)) / n

        # Store the error and computation time
        E_r.append(d)
        t_c.append(time.time() - iter_start_time)

    elapsed_time = time.time() - start_time
    print(f"ICP completed in {j} iterations and {elapsed_time:.2f} seconds")

    R_opt = R
    T_opt = T

    return R_opt, T_opt, E_r, t_c, elapsed_time

def ShowData(P, Q):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Point Cloud P (Original)'
    ))

    fig.add_trace(go.Scatter3d(
        x=Q[:, 0], y=Q[:, 1], z=Q[:, 2],
        mode='markers',
        marker=dict(size=2, color='red'),
        name='Point Cloud Q (Transformed)'
    ))

    fig.update_layout(
        title="3D Point Cloud Visualization",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )

    fig.show()

def MatrixComparer(R_A, R_B, T_A, T_B, t_a, t_b, err_a, err_b, method_A, method_B):
    R_A = R_A  
    T_A = T_A

    R_B = R_B 
    T_B = T_B

    t_a = t_a 
    t_b = t_b   

    R_difference =np.abs(R_B - R_A)
    T_difference =np.abs(T_B - T_A)
    time_difference =np.abs(t_b - t_a)
    err_difference=np.abs(err_b-err_a)


    def format_scientific(arr):
        return [np.format_float_scientific(val, precision=2) for val in arr]

    R_A_formatted = format_scientific(R_A.flatten())
    R_B_formatted = format_scientific(R_B.flatten())
    R_difference_formatted = format_scientific(R_difference.flatten())

    T_A_formatted = format_scientific(T_A)
    T_B_formatted = format_scientific(T_B)
    T_difference_formatted = format_scientific(T_difference)

    t_a_formatted = np.format_float_scientific(t_a, precision=2)
    t_b_formatted = np.format_float_scientific(t_b, precision=2)
    time_difference_formatted = np.format_float_scientific(time_difference, precision=2)

    err_a_formatted = np.format_float_scientific(err_a, precision=2)
    err_b_formatted = np.format_float_scientific(err_b, precision=2)
    err_difference_formatted = np.format_float_scientific(err_difference, precision=2)

    rotation_table = go.Figure(data=[go.Table(
        header=dict(values=["Element", method_A, method_B, "Difference Abs"],
                    fill_color="rgb(128,39,108)", font=dict(color="white"),
                    align='left'),
        cells=dict(values=[
            [f"R[{i},{j}]" for i in range(3) for j in range(3)],  
            R_A_formatted, 
            R_B_formatted,    
            R_difference_formatted     
        ],
        fill_color='lavender',
        align='left')
    )])

    translation_table = go.Figure(data=[go.Table(
        header=dict(values=["Component", method_A, method_B, "Difference Abs"],
                    fill_color="rgb(128,39,108)", font=dict(color="white"),
                    align='left'),
        cells=dict(values=[
            ["Tx", "Ty", "Tz"], 
            T_A_formatted,
            T_B_formatted,     
            T_difference_formatted     
        ],
        fill_color='lavender',
        align='left')
    )])

    time_table = go.Figure(data=[go.Table(
        header=dict(values=["Metric", method_A, method_B, "Difference Abs (s)"],
                    fill_color="rgb(128,39,108)", font=dict(color="white"),
                    align='left'),
        cells=dict(values=[
            ["Time"],  
            [t_a_formatted],  
            [t_b_formatted],      
            [time_difference_formatted]  
        ],
        fill_color='lavender',
        align='left')
    )])

    error_table = go.Figure(data=[go.Table(
        header=dict(values=["Metric", method_A, method_B, "Difference Abs"],
                    fill_color="rgb(128,39,108)", font=dict(color="white"),
                    align='left'),
        cells=dict(values=[
            ["Error"],  
            [err_a_formatted],  
            [err_b_formatted],      
            [err_difference_formatted]  
        ],
        fill_color='lavender',
        align='left')
    )])


    print("Rotation Matrix Comparison:")
    rotation_table.show()

    print("Translation Vector Comparison:")
    translation_table.show()

    print("Time Comparison:")
    time_table.show()

    print("Error Comparison:")
    error_table.show()


