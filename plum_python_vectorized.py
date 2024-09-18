import numpy as np
import scipy

def extract_degree_sequence(M):
    """
    Computes the degrees of the nodes of the plumbing graph given a plumbing matrix
    """
    # Use NumPy to sum the rows, excluding the diagonal elements
    degree_sequence = np.sum(M, axis=1) - np.diag(M)
    return degree_sequence

def extract_weight_vector(M):
    """
    Computes the weights of the nodes of a plumbing graph given the plumbing matrix
    """
    # Use NumPy to extract the diagonal elements
    weight_sequence = np.diag(M)
    return weight_sequence

def char_square(M_inv, k):
    """
    Computes the quadratic form k^T * M_inv * k
    """
    k = np.array(k)
    square = np.dot(k.T, np.dot(M_inv, k))
    return square

def zhat_vec(M,s_rep,n):
    # Set up data
    s = M.shape[0]
    s_rep = np.array(s_rep)
    degree_vector = extract_degree_sequence(M)
    weight_vector = extract_weight_vector(M)


    # Assert that the spin-c structure is congruent to the degree vector mod 2
    for i in range(s):
        if (float(s_rep[i])-float(degree_vector[i])) % 2 != 0:
            raise Exception("Your selected spin^c convention is to"
                            " use vectors congruent mod 2 to the"
                            " degree vector, but second parameter"
                            " does not satisfy this.")


    # Compute the min level
    M_inv = np.linalg.inv(M)
    s_rep_T = np.transpose(s_rep)
    a_quadratic_form = np.matmul(np.matmul(s_rep_T, M_inv), s_rep)
    min_level = (a_quadratic_form)/4
    
    # Compute the normalization term
    k = s_rep + weight_vector + degree_vector 
    k_squared = char_square(M_inv, np.transpose(k))
    normalization_term = -(k_squared + 3*s
                            + sum(weight_vector))/4\
                            + sum(k)/2\
                            - sum(weight_vector + degree_vector)/4

    # Calculate x based on the condition
    x = 2 * np.sqrt(np.abs(weight_vector) * (n - 1)) \
        if min_level % 1 == 0 else \
        2 * np.sqrt(np.abs(weight_vector) * n)

    # Create the bounding box array
    bounding_box = np.column_stack((-x, x)).tolist()

    # Build F_supp, this cannot be vectorized because F_supp is not a regular shape
    F_supp = []
    for i in range(s):
        values = list()
        if degree_vector[i] == 0:
            if bounding_box[i][0] <= -2 and bounding_box[i][1] >= 2: # I
                values = [-2,0,2]
                #F_supp.append([-2, 0, 2])
            elif bounding_box[i][0] <= -2 and bounding_box[i][1] < 2: # II
                values = [-2,0]
                #F_supp.append([-2, 0])
            elif bounding_box[i][0] > -2 and bounding_box[i][1] >= 2: # III
                values = [0,2]
                #F_supp.append([0, 2])
            else:
                values = [0]
                #F_supp.append([0])
        elif degree_vector[i] == 1:
            if bounding_box[i][0] <= -1 and bounding_box[i][1] >= 1: # I
                values = [-1, 1]
                #F_supp.append([-1, 1])
            elif bounding_box[i][0] <= -1 and bounding_box[i][1] < 1: # II
                values = [-1]
                #F_supp.append([-1])
            elif bounding_box[i][0] > -1 and bounding_box[i][1] >= 1: # III
                values = [1]
                #F_supp.append([1])
        elif degree_vector[i] == 2:
            values = [0]
            #F_supp.append([0])
        else:
            r = degree_vector[i]-2
            if bounding_box[i][0] <= -r: 
                values.append(-r)
                for j in range(1, int(np.floor((-r-bounding_box[i][0])/2))+1):
                    values.append(-r - 2*j)
                values.append(r)
                for j in range(1, int(np.floor((bounding_box[i][1]-r)/2))+1):
                    values.append(r + 2*j)
        F_supp.append(np.array(values))

    # Take products of F_supp
    degree_vector = np.array(degree_vector)
    M_inv = np.linalg.inv(M)
    #arrays = [ np.array(lst) for lst in F_supp]
    grid = np.meshgrid(*F_supp)
    y_arr = np.array([g.ravel() for g in grid]).T

    # Establish condition
    c_arr = -1*np.einsum("ij,jk,ik->i", y_arr, M_inv, y_arr)/4
    condition = np.all(np.mod(np.around(1/2 * ((y_arr - s_rep) @ M_inv),4),1) == 0.0, axis=1)

    # Impose condition
    c_arr_c = c_arr[condition]
    y_arr_c = y_arr[condition]

    # Compute F
    F = np.ones(y_arr_c.shape)
    mask_0 = degree_vector == 0
    F[:,mask_0] = np.where(y_arr_c[:,mask_0] == 0, -2*F[:,mask_0], 1)
    mask_1 = degree_vector == 1
    F[:,mask_1] = np.where(y_arr_c[:,mask_1] == 1, -F[:,mask_1], 1)
    mask_g2 = degree_vector > 2
    F[:,mask_g2] = np.where(np.abs(y_arr_c[:,mask_g2]) >= degree_vector[mask_g2]-2, 
                            F[:,mask_g2]*(1/2)*(np.sign(y_arr_c[:,mask_g2])**degree_vector[mask_g2]) *\
                            scipy.special.comb((degree_vector[mask_g2] + np.abs(y_arr_c[:,mask_g2]))/2 - 2,degree_vector[mask_g2]-3 ),
                            1)
    F = np.prod(F,axis=1)

    # Tally up the results
    c_arr_floor = np.floor(c_arr_c).astype(int) 
    c_unique = np.arange(n)
    F_sums = np.bincount(c_arr_floor, weights=F, minlength=n)
    zhat_powers = c_unique + np.ceil(min_level) + normalization_term
    # Build zhat_list
    zhat_list = list(zip(F_sums, zhat_powers))
    return zhat_list


if __name__ == "__main__":
    M = np.array([[-3, 1, 0, 0, 1, 0, 0], 
                [1, -4, 1, 1, 0, 0, 0], 
                [0, 1, -1, 0, 0, 0, 0], 
                [0, 1, 0, -4, 0, 0, 0], 
                [1, 0, 0, 0, -1, 1, 1], 
                [0, 0, 0, 0, 1, -4, 0], 
                [0, 0, 0, 0, 1, 0, -3]])
    degree_sequence = extract_degree_sequence(M)
    z_hat_list = zhat_vec(M, degree_sequence, 30000)
    for tuple in z_hat_list:
        if(tuple[0] != 0):
            print('{c}q^{e}'.format(c=tuple[0], e=tuple[1]))
