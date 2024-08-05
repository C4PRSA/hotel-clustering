import numpy as np
from scipy.interpolate import CubicSpline, PPoly
import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
import distance_functions
from scipy.optimize import minimize
import math

##############
### second try

def updated_distance_fxn(spline, point):
    def helper(xi):
        if not isinstance(xi, (np.ndarray, list, tuple)):
            return math.dist((xi, spline(xi)), point)
        else:
            out = []
            for x in xi:
                out.append(math.dist((x, spline(x)), point))
            return out
    return helper


def reformat_data(data, highway_spline, decimals = 4, method = "derivative"):
    """
    Takes data of at least x, y and adds smallest distance of each point to highway with given coefficients.
    """
    r_three = []
    for data_point in data:
        r_three.append([create_third_dimension(highway_spline, data_point)])
    np.set_printoptions(suppress=True, precision = decimals)
    return np.append(data, r_three, axis=1)



def find_minimum_distance(spline_est, full_point):
    """
    Returns x, y, distance of closest point on a highway.
    """
    point = (full_point[0], full_point[1])
    new_min = minimize(updated_distance_fxn(spline_est, point), point[0])
    all_guesses = new_min.x
    if len(all_guesses) == 0:
        print("No minimums")
        return
    if len(all_guesses) == 1:
        xi = all_guesses[0]
        # print((xi, float(spline_est(xi))))
        # print(point)
        return xi, float(spline_est(xi)), math.dist((xi, float(spline_est(xi))), point)

    xi  = all_guesses[0]
    final_min = math.dist((xi, spline_est(xi)), point)
    for x in all_guesses:
        if math.dist((x, spline_est(x)), point) < final_min:
         xi = x
    return xi, float(spline_est(xi)), final_min


def signed_RHS_LHS_spline(point, spline, hw_x, hw_y, distance):
    """
        Uses cross product to determine if point is on the RHS or LHS of a given highway.
    """
    x = [1, spline.derivative()(hw_x), 0]
    y = [point[0] - hw_x, point[1] - hw_y, 0]
    c = np.cross(x, y)
    cross_product = c[2]
    if cross_product > 0:
        return distance
    return -distance

def create_third_dimension(spline, point):
    x, y, min_dist = find_minimum_distance(spline, point)
    return signed_RHS_LHS_spline(point, spline, x, y, min_dist)


def retrieve_hotel_data(file_name):
    """ Reads txt file that contains hotel data.
        Assumes in columns of Name, X location, Y location, and Rating."""
    file = open(file_name, "r")
    file.readline()
    data = []
    for line in file.readlines():
        data.append(np.array([float(d) for d in line.split()[1:4]]))
    return np.array(data)

#################################################################
### Module Testing
#################################################################

# x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
# y = [2.644,3.249,3.951,4.484,4.868,5.184,5.462,5.648,5.716,5.657,5.545,5.437,5.323,5.301,5.224,5.137,4.998,4.916,4.734,4.619,4.244,3.229,2.254,1.942,1.667,1.234,0.874,0.658]

# spline = CubicSpline(x, y)

# hw_x = np.linspace(0, 28, num=100)
# plotting_fxn = spline(hw_x)

# # geo_x = [0.0, 0.6, -0.7, 0.1]
# # geo_y = [0.7, -0.4, 0.0, -0.6]
# # att_x = [0.2, 0.5, 0.8, 1.4]
# # att_y = [1.4, 1.0, 0.6, 0.3]
# # centers_list = list(zip(geo_x, geo_y, att_x, att_y))

# # data_vectors, data_labels, centers = make_blobs(n_features = 2, n_samples = 40, cluster_std=0.35, centers=centers_list, return_centers=True)

# data_vectors = retrieve_hotel_data("hotel_data_manual.txt")
# # # highway_parameters = [0.1, -0.9, 2.3, 2.5]
# # highway_parameters = [2.5, 2.3, -0.9, 0.1]

# # test_point = (10, 10)
# pnt = (10, 2)

# # third_dem = create_third_dimension(highway_parameters, test_point)

# # print(third_dem)

# # highway_fxn = np.polynomial.Polynomial(highway_parameters)
# # highway_x, highway_y = highway_fxn.linspace(domain=(-1.5, 1.5))
# data = reformat_data(data_vectors, spline)
# print("******data_vectors**********")
# print(data_vectors)

# print("******data_vectors NEW**********")
# print(data)


# def plot_hw_and_hotels(hotel_data, spline_est, title=""):
#     """
#     Plots x and y location of hotels in given data, with color gradient representing rating.
#     """
#     hw_x = np.linspace(0, 28, num=100)
#     plotting_fxn = spline_est(hw_x)

#     plt.plot(hw_x, plotting_fxn)
#     temp_x = hotel_data[:, 0]
#     temp_y = hotel_data[:, 1]
#     # plt.scatter(temp_x, temp_y, c=Normalize(clip=False)(data[:, 2]), cmap="viridis")
#     plt.scatter(temp_x, temp_y, c=data[:, 2], cmap="viridis")
#     plt.legend(["highway", "hotels"])
#     plt.colorbar(label="hotel rating")
#     # plt.xlim((min(hw_x), max(hw_x)))
#     # plt.ylim((min(hw_y), max(hw_y)))
#     plt.title(title)

# plot_hw_and_hotels(data, spline, title="Distance with Spline format")
# plt.show()
