import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt


# -------------------------------------------- GENERATION OF RSSI POINTS -----------------------------------------------


def generate_rssi_points(file_path, num_points=1000):
    # Gateway position and parameters for path loss model
    gateway_position = (2500, 2500)
    pi = 14
    PL_d0 = 82
    gamma = 2

    # Generate random (x, y) coordinates and calculate RSSI
    data = {
        'x': [],
        'y': [],
        'RSSI': []
    }

    for _ in range(num_points):
        x_coord = random.uniform(0, 5000)
        y_coord = random.uniform(0, 5000)

        # Calculate RSSI based on the distance from the gateway
        distance = np.sqrt((x_coord - gateway_position[0]) ** 2 + (y_coord - gateway_position[1]) ** 2)
        RSSI = pi - PL_d0 - 10 * gamma * np.log10(distance / 1)

        # Store the generated data
        data['x'].append(x_coord)
        data['y'].append(y_coord)
        data['RSSI'].append(RSSI)

    # Save the data to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Generated RSSI points saved to {file_path}")


# -------------------------------------------- GENERATION OF TRAJECTORY -----------------------------------------------

def generate_trajectory(file_path, num_points=100, velocity_range=(0.5, 5.0)):
    # Define circular trajectory center and radius
    center = (2500, 2500)
    radius = 1000

    # Time step between each point
    time_step = 10  # seconds

    data = {
        'X Coordinate': [],
        'Y Coordinate': [],
        'Speed (m/s)': []
    }

    # Initial point on the circle
    angle = 0
    for _ in range(num_points):
        # Calculate (x, y) coordinates for the current angle
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)

        # Randomly assign a velocity for each point
        speed = random.uniform(velocity_range[0], velocity_range[1])

        # Store the generated data
        data['X Coordinate'].append(x)
        data['Y Coordinate'].append(y)
        data['Speed (m/s)'].append(speed)

        # Increment the angle for the next point
        angle += speed * time_step / radius

    # Save the trajectory to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Generated trajectory saved to {file_path}")


# -------------------------------------------- HELPER FUNCTIONS -----------------------------------------------

# Function to calculate the next point in a circular trajectory
def next_point_on_circle(current_point, center, radius, velocity, time_step):
    dx = current_point[0] - center[0]
    dy = current_point[1] - center[1]
    current_angle = np.arctan2(dy, dx)

    omega = velocity / radius
    angle_step = omega * time_step
    next_angle = current_angle + angle_step

    next_x = center[0] + radius * np.cos(next_angle)
    next_y = center[1] + radius * np.sin(next_angle)

    return np.array([next_x, next_y])


# Function to calculate the radius of curvature and center of the trajectory
def radius_of_curvature(A, B, C):
    MAB = (A + B) / 2
    MBC = (B + C) / 2

    delta_AB = B - A
    delta_BC = C - B
    gradient_AB = -delta_AB[0] / delta_AB[1] if delta_AB[1] != 0 else np.inf
    gradient_BC = -delta_BC[0] / delta_BC[1] if delta_BC[1] != 0 else np.inf

    def bisector(m, midpoint):
        if m == np.inf:
            return lambda x: midpoint[1]
        else:
            return lambda x: m * (x - midpoint[0]) + midpoint[1]

    bisector_AB = bisector(gradient_AB, MAB)
    bisector_BC = bisector(gradient_BC, MBC)

    x_center = (MBC[1] - MAB[1] + gradient_AB * MAB[0] - gradient_BC * MBC[0]) / (gradient_AB - gradient_BC)
    y_center = bisector_AB(x_center)

    center = np.array([x_center, y_center])
    radius = np.linalg.norm(center - A)

    return radius, center


# -------------------------------------------- CLUSTERING RSSI POINTS -----------------------------------------------

def perform_rssi_clustering(rssi_file_path, num_clusters=10):
    # Load RSSI points from the CSV file
    data = pd.read_csv(rssi_file_path)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data[['x', 'y']])
    data['cluster'] = kmeans.labels_

    # Visualize the clusters
    plt.scatter(data['x'], data['y'], c=data['cluster'], cmap='viridis')
    plt.title(f'KMeans Clustering of RSSI Points (k={num_clusters})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    return data


# -------------------------------------------- MAIN FUNCTIONS -----------------------------------------------

# RSSI Prediction using KNN
def predict_rssi_using_knn(trajectory_file, rssi_data_file, k=3):
    data = pd.read_csv(rssi_data_file)
    X_train = data[['x', 'y']]
    y_train = data['RSSI']

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    trajectory = pd.read_csv(trajectory_file)

    pi = 14
    PL_d0 = 82
    gamma = 2
    gateway_position = (2500, 2500)

    trajectory['RSSI_meas'] = None
    trajectory['RSSI_predict'] = None

    for index, row in trajectory.iterrows():
        x_coord = row['X Coordinate']
        y_coord = row['Y Coordinate']

        predicted_snr = knn.predict([[x_coord, y_coord]])

        dik = np.sqrt((x_coord - gateway_position[0]) ** 2 + (y_coord - gateway_position[1]) ** 2)
        zik = pi - PL_d0 - 10 * gamma * np.log10(dik / 1)
        RSSI_measured = zik - 0  # Nik is 0 in this case

        trajectory.at[index, 'RSSI_meas'] = RSSI_measured
        trajectory.at[index, 'RSSI_predict'] = predicted_snr[0]

    output_file = 'src/evaluation/scenario_3/data/k_predictions_trajectory.csv'
    trajectory.to_csv(output_file, index=False)

    print(f"RSSI predictions saved to {output_file}")


# Trajectory Estimation based on circular motion
def estimate_trajectory(trajectory_file):
    trajectory = pd.read_csv(trajectory_file)

    for i in range(len(trajectory) - 3):
        A = trajectory.loc[i, ['X Coordinate', 'Y Coordinate']].values
        B = trajectory.loc[i + 1, ['X Coordinate', 'Y Coordinate']].values
        C = trajectory.loc[i + 2, ['X Coordinate', 'Y Coordinate']].values

        radius, center = radius_of_curvature(A, B, C)

        velocity = trajectory.loc[i + 3, 'Speed (m/s)']
        time_step = 10  # seconds

        next_point = next_point_on_circle(C, center, radius, velocity, time_step)
        print(f"Next point in trajectory: {next_point}")


# -------------------------------------------- MAIN EXECUTION -----------------------------------------------

if __name__ == "__main__":
    # File paths
    rssi_file_path = "src/evaluation/scenario_3/data/rssi_points/rssi_points_1000.csv"
    trajectory_file_path = "trajectory.csv"

    # Generate RSSI points and trajectory
    generate_rssi_points(rssi_file_path, num_points=1000)
    generate_trajectory(trajectory_file_path, num_points=100)

    # Perform clustering on RSSI points
    clustered_data = perform_rssi_clustering(rssi_file_path, num_clusters=10)

    # Predict RSSI for the generated trajectory
    predict_rssi_using_knn(trajectory_file_path, rssi_file_path, k=5)

    # Estimate next points on the trajectory based on circular motion
    estimate_trajectory(trajectory_file_path)
