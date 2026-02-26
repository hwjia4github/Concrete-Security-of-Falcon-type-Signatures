from math import *
import numpy as np
from mpmath import mp
mp.dps = 700
import falcon

def delta_beta(b):
    return (((pi * b) ** (1. / b) * b) / (2 * pi * exp(1))) ** (0.5 / (b - 1))

def smooth(logeps, m):
    return sqrt(log(2 * m) + logeps * log(2)) / sqrt(2 * pi * pi)

def anti_smooth(r, dim):
    # print(r)
    left_logeps = 1
    right_logeps = 3000
    mid_logeps = (left_logeps + right_logeps) / 2
    t = smooth(mid_logeps, dim)
    while abs(t - r) > 10 ** (-15):
        if t > r:
            right_logeps = mid_logeps
        else:
            left_logeps = mid_logeps
        mid_logeps = (left_logeps + right_logeps) / 2
        t = smooth(mid_logeps, dim)
    return mid_logeps

def lwe_attack(n, Q, sigma_f):
    best_b = 9999
    best_k = 0
    for k in range(0, n):
        dim = 2 * n - k
        for beta in range(200, 1500):
            delta = delta_beta(beta)
            LHS = sigma_f * sqrt(beta) * sqrt(3. / 4)
            RHS = delta ** (2. * beta - dim - 1) * Q ** (1. * (n - k) / dim)
            if LHS < RHS:
                if best_b > beta:
                    best_b = beta
                    best_k = k
                break
    return 0.292 * best_b

def sis_attack(n, m, Q, sig_norm):
    min_beta = 5000
    the_k = 0
    for k in range(m - n):
        dim = m - k
        for beta in range(100, 5000):
            delta = delta_beta(beta)
            Vol_root = Q ** (n / dim)
            LHS = (delta ** dim) * Vol_root
            RHS = sig_norm
            if RHS > LHS:
                if min_beta > beta:
                    min_beta = beta
                    the_k = k
                break
    #print('best_m: ', m - the_k)
    #return min_beta, 0.292 * min_beta, 0.265 * min_beta
    return 0.292 * min_beta

def preimage_size(n, std):
    """ Compute the storage size (in byte) of dim-n Gaussian vector with standard deviation std
    Args:
        n: dimension
        std: standard deviation of Gaussian
    Returns:
        number of bytes to store dim-n Gaussian vector
    """
    entropy = ceil((0.5 + log(sqrt(2 * pi) * std)) / log(2) * n)
    return entropy / 8

def collect_leaves(node):

    leaves = []
    if isinstance(node, list):
        # 检查是否是叶子节点（包含两个元素，第二个是0）
        if len(node) == 2 and node[1] == 0:
            leaves.append(node[0]) 
        else:
            # 递归处理子节点 
            for item in node:
                leaves.extend(collect_leaves(item)) 
    return leaves

def comp_re_p_ours(logeps_data): # relative error for sampler
    eps = mp.mpf(1.0)
    dim = len(logeps_data)
    for i in range(0, dim):
        eps = eps * (mp.mpf(1.0) + mp.power(mp.mpf(2.0), mp.mpf(-logeps_data[i])))
    return eps ** 2 - 1.0

def comp_re_p_falcon(logeps): # relative error for falcon
    return mp.mpf(2.0) * mp.power(mp.mpf(2.0), mp.mpf(-logeps))
 
def comp_renyi_divergence(a, re):
    return mp.mpf(1.0) + a * mp.power(re, mp.mpf(2)) / mp.mpf(2)

def optimal_renyi_order(Q, re, lam):
    k = mp.power(re, 2) / 2
    m = Q * k / mp.log(2)
    a1 = (lam * k + mp.sqrt(lam * k * lam * k + mp.mpf(4) * lam * m)) / mp.mpf(2) / m
    #a2 = (lam * k - mp.sqrt(lam * k * lam * k + mp.mpf(4) * lam * m)) / mp.mpf(2) / m
    return a1

def reduced_security(Q, a, re, lam):
    rd = mp.mpf(1.0) + a * mp.power(re, mp.mpf(2)) / mp.mpf(2) # compute renyi_divergence
    return Q * mp.log(rd, 2) + lam / a

def comp_ours_uf(n):
    r_data = gen_r_date(n)
    logeps_data = []
    for i in range(0, n):
        logeps_data.append(anti_smooth(r_data[i], 1)) 

    Q_s = mp.mpf(2 ** 64)
    if n == 512:
        lam = 120 
        C_s = Q_s + mp.mpf(2 ** 50)
        logeps = 35.5
    elif n == 1024:
        lam = 278
        C_s = Q_s + mp.mpf(2 ** 36)
        logeps = 36

    re_p = comp_re_p_ours(logeps_data)
    a_p = optimal_renyi_order(C_s, re_p, lam)
    red_sec_p = reduced_security(C_s, a_p, re_p, lam)
    lam = mp.power(mp.mpf(lam - red_sec_p), mp.mpf(a_p - 1) / mp.mpf(a_p))

    # naive method
    epsilon = mp.power(mp.mpf(2.0), mp.mpf(-logeps))
    # logepsprime = anti_smooth(anti_smooth(smooth(logeps, 2 * n) * 1.17 * sqrt(12289))) # logepsprime = 782250, too small
    re_u = mp.mpf(1.0) * epsilon / (mp.mpf(1.0) - epsilon)
    a_u = optimal_renyi_order(C_s, re_u, lam)
    red_sec_u = reduced_security(C_s, a_u, re_u, lam)

    # new method
    r = sqrt(2) * sqrt(log(4 * n * (1. + mp.mpf(1.0) / epsilon)) / 2) / pi
    newlogeps = anti_smooth(r, 2 * n)
    # print(newlogeps)
    re_u2 = mp.power(mp.mpf(2.0), mp.mpf(-newlogeps / 2))
    a_u2 = optimal_renyi_order(C_s, re_u2, lam)
    red_sec_u2 = reduced_security(C_s, a_u2, re_u2, lam)

    print("ours-" + str(n) + "-uf: optimal_renyi_order for sampler and unifrom: ", (round(a_p, 2), round(a_u, 2), round(a_u2, 2)))
    print("ours-" + str(n) + "-uf: reduced_security for sampler and unifrom:", (round(red_sec_p, 3), round(red_sec_u, 3), round(red_sec_u2, 3)))
    print("")

def comp_falcon_uf(n):
    Q_s = mp.mpf(2 ** 64)
    if n == 512:
        lam = 120 # mp.mpf(sis_security)
        C_s = Q_s + mp.mpf(2 ** 50)
        logeps = 35.5
    elif n == 1024:
        lam = 278
        C_s = Q_s + mp.mpf(2 ** 36)
        logeps = 36

    re_p = comp_re_p_falcon(logeps)
    a_p = optimal_renyi_order(C_s, re_p, lam) 
    red_sec_p = reduced_security(C_s, a_p, re_p, lam)
    lam = mp.power(mp.mpf(lam - red_sec_p), mp.mpf(a_p - 1) / mp.mpf(a_p))

    epsilon = mp.power(mp.mpf(2.0), mp.mpf(-logeps))
    re_u = mp.mpf(2.0) * epsilon / (mp.mpf(1.0) - epsilon)
    a_u = optimal_renyi_order(C_s, re_u, lam)
    red_sec_u = reduced_security(C_s, a_u, re_u, lam)

    print("Falcon-" + str(n) + "-uf: optimal_renyi_order for sampler and unifrom: ", (round(a_p, 2), round(a_u, 2)))
    print("Falcon-" + str(n) + "-uf: reduced_security for sampler and unifrom: ", (round(red_sec_p, 2), round(red_sec_u, 2)))
    print("")

def comp_falcon_suf(n):
    Q_s = mp.mpf(2 ** 64)
    q = 12289
    if n == 512:
        # lam = sis_attack(n, 2 * n, q, sqrt(2) * beta) # mp.mpf(sis_security)
        C_s = Q_s + mp.mpf(2 ** 50)
        logeps = 35.5
    elif n == 1024:
        # lam = lam = sis_attack(n, 2 * n, q, sqrt(2) * 168.388571447)#278
        C_s = Q_s + mp.mpf(2 ** 36)
        logeps = 36

    r = smooth(logeps, 2 * n)
    sigma = 1.17 * sqrt(q) * r
    beta = 1.1 * sigma * sqrt(2 * n)
    lam = sis_attack(n, 2 * n, q, sqrt(2) * beta) 
    print("initial suf-cma security of falcon: ", lam)

    re_p = comp_re_p_falcon(logeps)
    a_p = optimal_renyi_order(C_s, re_p, lam) 
    red_sec_p = reduced_security(C_s, a_p, re_p, lam)
    lam = mp.power(mp.mpf(lam - red_sec_p), mp.mpf(a_p - 1) / mp.mpf(a_p))

    Q = C_s + mp.mpf(2 ** 60)
    epsilon = mp.power(mp.mpf(2.0), mp.mpf(-logeps))
    re_u = mp.mpf(2.0) * epsilon / (mp.mpf(1.0) - epsilon)
    a_u = optimal_renyi_order(Q, re_u, lam)
    red_sec_u = reduced_security(Q, a_u, re_u, lam)

    # rd_p = mp.mpf(1.0) + a_p * mp.power(re_p, mp.mpf(2)) / mp.mpf(2)
    # rd_u = mp.mpf(1.0) + a_u * mp.power(re_u, mp.mpf(2)) / mp.mpf(2)
    # L = mp.power(2, mp.mpf(60))
    # term1 = mp.power(2, mp.mpf(96-60)) * mp.power(rd_u, Q) * mp.power(rd_p, C_s)
    # print(log(term1, 2))

    # print("Falcon-" + str(n) + "-suf: optimal_renyi_order for sampler and unifrom: ", (round(a_p, 2), round(a_u, 2)))
    # print("Falcon-" + str(n) + "-suf: reduced_security for sampler and unifrom: ", (round(red_sec_p, 2), round(red_sec_u, 2)))
    # print("")

def comp_ours_suf(n):
    r_data = gen_r_date(n)
    logeps_data = []
    for i in range(0, n):
        logeps_data.append(anti_smooth(r_data[i], 1)) 

    Q_s = mp.mpf(2 ** 64)
    if n == 512:
        lam = 95
        C_s = Q_s + mp.mpf(2 ** 50)
        logeps = 35.5
    elif n == 1024:
        lam = 278
        C_s = Q_s + mp.mpf(2 ** 36)
        logeps = 36

    re_p = comp_re_p_ours(logeps_data)
    a_p = optimal_renyi_order(C_s, re_p, lam)
    red_sec_p = reduced_security(C_s, a_p, re_p, lam)
    lam = mp.power(mp.mpf(lam - red_sec_p), mp.mpf(a_p - 1) / mp.mpf(a_p))

    # naive method
    epsilon = mp.power(mp.mpf(2.0), mp.mpf(-logeps))
    # logepsprime = anti_smooth(anti_smooth(smooth(logeps, 2 * n) * 1.17 * sqrt(12289))) # logepsprime = 782250, too small
    re_u = mp.mpf(1.0) * epsilon / (mp.mpf(1.0) - epsilon)
    a_u = optimal_renyi_order(C_s, re_u, lam)
    red_sec_u = reduced_security(C_s, a_u, re_u, lam)

    # new method
    r = sqrt(2) * sqrt(log(4 * n * (1. + mp.mpf(1.0) / epsilon)) / 2) / pi
    newlogeps = anti_smooth(r, 2 * n)
    # print(newlogeps)
    re_u2 = mp.power(mp.mpf(2.0), mp.mpf(-newlogeps / 2))
    a_u2 = optimal_renyi_order(C_s, re_u2, lam)
    red_sec_u2 = reduced_security(C_s, a_u2, re_u2, lam)

    print("ours-" + str(n) + "-suf: optimal_renyi_order for sampler and unifrom: ", (round(a_p, 2), round(a_u, 2), round(a_u2, 2)))
    print("ours-" + str(n) + "-suf: reduced_security for sampler and unifrom:", (round(red_sec_p, 3), round(red_sec_u, 3), round(red_sec_u2, 3)))
    print("")

def find_opt_logeps_p(n, the_logeps=None): # not good
    r_data = gen_r_date(n)
    logeps_data = []
    for i in range(0, n):
        logeps_data.append(anti_smooth(r_data[i], 1)) 

    Q_s = mp.mpf(2 ** 64)
    if n == 512:
        lam = 120 
        C_s = Q_s + mp.mpf(2 ** 50)
        logeps = 35.5
    elif n == 1024:
        lam = 278
        C_s = Q_s + mp.mpf(2 ** 36)
        logeps = 36

    if the_logeps != None:
        scalar = smooth(the_logeps, 2 * n) / smooth(logeps, 2 * n)
        r_data = [x * scalar for x in r_data]
        logeps_data = []
        for i in range(0, n):
            logeps_data.append(anti_smooth(r_data[i], 1))
        
        re_p = comp_re_p_ours(logeps_data)
        a_p = optimal_renyi_order(C_s, re_p, lam)
        red_sec_p = reduced_security(C_s, a_p, re_p, lam)
        print("renyi_order %.2f and reduced security %.2f for sampler with given logeps = %.2f: " %(round(a_p, 2), round(red_sec_p, 2), the_logeps))
        return 

    for the_logeps in range(35, 20, -1):
        scalar = smooth(the_logeps, 2 * n) / smooth(logeps, 2 * n)
        r_data = [x * scalar for x in r_data]
        logeps_data = []
        for i in range(0, n):
            logeps_data.append(anti_smooth(r_data[i], 1))
        
        re_p = comp_re_p_ours(logeps_data)
        a_p = optimal_renyi_order(C_s, re_p, lam)
        red_sec_p = reduced_security(C_s, a_p, re_p, lam)

        print("optimal_renyi_order %.2f and reduced security %.2f for sampler with opt. logeps = %.2f: " %(round(a_p, 2), round(red_sec_p, 2), the_logeps))
        if red_sec_p > 1:
            break 


def gen_r_date(n):
    sk = falcon.SecretKey(n)   # pk = falcon.PublicKey(sk)
    tree_data = sk.T_fft 
    r_data = collect_leaves(tree_data)
    return r_data

def Base_eps_sigma(n, iter, sigma, factor=1):
    '''
    n: poly dim
    iter: compute the first iter cols of B_data
    sigma: stdev
    factor: scale the GSO vectors
    '''
    filename = f"B_data_{n}_ssh.npy"
    loaded = np.load(filename)
    s_p = mp.mpf(sigma) * mp.sqrt(mp.mpf(2 * pi))
    # iter  = 10
    eps_list = np.zeros(iter + 1)
    # sum = mp.mpf(0)
    for j in range(iter, iter + 1):
        sum = mp.mpf(0)
        # if j%300 == 0:
        #     print("j",j)
        for i in range(n):
            sum = sum + mp.mpf(4.0) * mp.exp(-mp.mpf(pi) * s_p * s_p / (mp.mpf(loaded[i][j]) * mp.mpf(loaded[i][j])) / factor / factor)
        eps = sum / (mp.mpf(1.0) - sum)
        eps_list[j]  = mp.ln(eps) / mp.ln(2.0)
        t = -eps_list[j]
    return t #eps_list


def print_falcon_statistics(n, CNT = 1000):
    filename = f"B_data_{n}_ssh.npy"
    loaded = np.load(filename)
    r_data = np.zeros(n)
    red_sec_p_data = []
    red_sec_u_data = []
    Q_s = mp.mpf(2 ** 64)

    if n == 512:
        lam = 120 
        sigma = 165.736617183
        C_s = Q_s + mp.mpf(2 ** 50)
    elif n == 1024:
        lam = 278
        sigma = 168.388571447
        C_s = Q_s + mp.mpf(2 ** 36)

    max_red_sec_p = 0
    max_red_sec_u = 0
    for cnt in range (0, CNT):
        #r_data = gen_r_date(n)
        for j in range(n):
            r_data[j] = sigma / loaded[j][cnt]

        logeps_data = []
        for i in range(0, n):
            logeps_data.append(anti_smooth(r_data[i], 1)) 

        re_p = comp_re_p_ours(logeps_data)
        a_p = optimal_renyi_order(C_s, re_p, lam)
        red_sec_p = reduced_security(C_s, a_p, re_p, lam)
        red_sec_p_data.append(red_sec_p)
        if red_sec_p > max_red_sec_p:
            max_a_p = a_p
            max_red_sec_p = red_sec_p

        # if cnt % 100 == 0: 
        #     print(cnt)
        lam2 = mp.power(mp.mpf(lam - max_red_sec_p), mp.mpf(a_p - 1) / mp.mpf(a_p))
        new_logeps = Base_eps_sigma(n, cnt, sigma)
        new_epsilon = mp.power(mp.mpf(2.0), mp.mpf(-new_logeps))
        re_u = new_epsilon / (mp.mpf(1.0) - new_epsilon)
        a_u = optimal_renyi_order(C_s, re_u, lam2)
        red_sec_u = reduced_security(C_s, a_u, re_u, lam2)
        red_sec_u_data.append(red_sec_u)
        if red_sec_u > max_red_sec_u:
            max_a_u = a_u
            max_red_sec_u = red_sec_u

    max_p, min_p, ave_p, var_p = round(np.max(red_sec_p_data), 10), round(np.min(red_sec_p_data), 10), round(np.average(red_sec_p_data), 10), round(np.var(red_sec_p_data), 10)
    max_u, min_u, ave_u, var_u = round(np.max(red_sec_u_data), 10), round(np.min(red_sec_u_data), 10), round(np.average(red_sec_u_data), 10), round(np.var(red_sec_u_data), 10)
    print(max_p, min_p, ave_p, var_p)
    print(max_u, min_u, ave_u, var_u)
    print((round(max_a_p, 5), round(max_red_sec_p, 5)), (round(max_a_u, 5), round(max_red_sec_u, 5)))


def find_opt_logeps_u_suf(n, the_logeps=None):
    # if the_logeps != None:
    #     epsilon = mp.power(mp.mpf(2.0), mp.mpf(-the_logeps))
    #     r = sqrt(2) * sqrt(log(4 * n * (1. + mp.mpf(1.0) / epsilon)) / 2) / pi
    #     newlogeps = anti_smooth(r, 2 * n)
    #     re_u = mp.power(mp.mpf(2.0), mp.mpf(-newlogeps / 2))
    #     a_u = optimal_renyi_order(C_s, re_u, lam)
    #     red_sec_u = reduced_security(C_s, a_u, re_u, lam)
    #     print("renyi_order %.2f and reduced security %.2f for sampler with given logeps = %.2f: " %(round(a_u, 2), round(red_sec_u, 2), the_logeps))
    #     return 

    if n == 512:
        C_s = mp.mpf(2 ** 64) + mp.mpf(2 ** 50)
    elif n == 1024:
        C_s = mp.mpf(2 ** 64) + mp.mpf(2 ** 36)
    

    for i in range(260, 240, -1):
        the_logeps = i / 10
        epsilon = mp.power(mp.mpf(2.0), mp.mpf(-the_logeps))
        r = sqrt(2) * sqrt(log(4 * n * (1. + mp.mpf(1.0) / epsilon)) / 2) / pi
        newlogeps = anti_smooth(r, 2 * n)
        re_u = mp.power(mp.mpf(2.0), mp.mpf(-newlogeps / 2))

        q = 12289
        sigma = 1.17 * sqrt(q) * smooth(the_logeps, 2 * n)
        beta = 1.1 * sigma * sqrt(2 * n)
        lam = sis_attack(n, 2 * n, q, sqrt(2) * beta)

        for the_l in range(40, 70, 1):           
            # x = comp_red_sec_u(re_u, lam, the_l)
            # y = comp_red_sec_u(re_u, lam, the_l + 1)
            # if x <= the_l and y > the_l:
            Q = C_s + mp.mpf(2 ** (the_l))
            a_u = optimal_renyi_order(Q, re_u, lam)
            red_sec_u = reduced_security(Q, a_u, re_u, lam)
            if red_sec_u <= the_l:
                print("renyi_order %.2f and reduced security %.2f for sampler with given logeps = %.2f and l = %.2d: " %(round(a_u, 2), round(red_sec_u, 2), the_logeps, the_l))
                break

def find_opt_logeps_u_uf(n, the_logeps=None):
    # if the_logeps != None:
    #     epsilon = mp.power(mp.mpf(2.0), mp.mpf(-the_logeps))
    #     r = sqrt(2) * sqrt(log(4 * n * (1. + mp.mpf(1.0) / epsilon)) / 2) / pi
    #     newlogeps = anti_smooth(r, 2 * n)
    #     re_u = mp.power(mp.mpf(2.0), mp.mpf(-newlogeps / 2))
    #     a_u = optimal_renyi_order(C_s, re_u, lam)
    #     red_sec_u = reduced_security(C_s, a_u, re_u, lam)
    #     print("renyi_order %.2f and reduced security %.2f for sampler with given logeps = %.2f: " %(round(a_u, 2), round(red_sec_u, 2), the_logeps))
    #     return 

    if n == 512:
        C_s = mp.mpf(2 ** 64) + mp.mpf(2 ** 50)
    elif n == 1024:
        C_s = mp.mpf(2 ** 64) + mp.mpf(2 ** 36)

    for the_logeps in range(250, 230, -1):
        the_logeps /= 10
        epsilon = mp.power(mp.mpf(2.0), mp.mpf(-the_logeps))
        r = sqrt(2) * sqrt(log(4 * n * (1. + mp.mpf(1.0) / epsilon)) / 2) / pi
        newlogeps = anti_smooth(r, 2 * n)
        re_u = mp.power(mp.mpf(2.0), mp.mpf(-newlogeps / 2))

        q = 12289
        sigma = 1.17 * sqrt(q) * smooth(the_logeps, 2 * n)
        beta = 1.1 * sigma * sqrt(2 * n)
        lam = sis_attack(n, 2 * n, q, beta)

        a_u = optimal_renyi_order(C_s, re_u, lam)
        red_sec_u = reduced_security(C_s, a_u, re_u, lam)
        if red_sec_u <= 96:
            print("lamda %.2f renyi_order %.2f and reduced security %.2f for simulator with logeps = %.2f: " %(round(lam, 2), round(a_u, 2), round(red_sec_u, 2), the_logeps))


def print_new_ws_falcon_uf_suf(n, q, logeps, beta_factor=1): # logeps = 20.6 is good for uf and 20.5 for suf
    sigma_f = 1.17 * sqrt(q / (2 * n))
    sigma_pre = 1.17 * sqrt(q) * smooth(logeps, 2 * n)
    tau = 1.1
    beta = tau * sigma_pre * sqrt(2 * n)
    norm_bound = beta_factor * beta

    key_rec = lwe_attack(n, q, sigma_f)
    forg_sec = sis_attack(n, 2 * n, q, norm_bound)

    sig_size = ceil(preimage_size(n, sigma_pre) + 40)

    print("sigma preimage: ", sigma_pre)
    print("norm bound", norm_bound)
    print("pk_size: ", n * (ceil(log(q) / log(2))) / 8)
    print("sig_size: ", sig_size)
    print("Key Recovery: ", key_rec)
    print("Forgery attack", forg_sec)

def opt_new_ws_falcon_uf_suf(n, q, logeps, beta_factor=1, COUNT=1000):
    basis_norm_factor = mp.sqrt(q) / mp.sqrt(12289)
    red_sec_u_data = []
    if n == 512:
        C_s = mp.mpf(2 ** 64) + mp.mpf(2 ** 50)
    elif n == 1024:
        C_s = mp.mpf(2 ** 64) + mp.mpf(2 ** 36)

    r = smooth(logeps, 2 * n)
    sigma = 1.17 * sqrt(q) * r
    beta = 1.1 * sigma * sqrt(2 * n)
    lam = sis_attack(n, 2 * n, q, beta_factor * beta)  #################################
    # print("initial suf-cma lambda of falcon: ", lam)
    lam_k = lwe_attack(n, q, sigma / sqrt(2 * n) / r)
    print("key / forg. sec, c*beta/q, |pk|: ", lam_k, lam, beta, beta_factor * beta / q, ceil(log(q, 2)) * n / 8)

    sigma = 1.17 * sqrt(q) * smooth(logeps, 2 * n)
    max_red_sec_u = 0
    for cnt in range(COUNT):
        new_logeps = Base_eps_sigma(n, cnt, sigma, basis_norm_factor)
        new_epsilon = mp.power(mp.mpf(2.0), mp.mpf(-new_logeps))
        
        r = sqrt(2) * sqrt(log(4 * n * (1. + mp.mpf(1.0) / new_epsilon)) / 2) / pi
        new_new_logeps = anti_smooth(r, 2 * n)
        re_u = mp.power(mp.mpf(2.0), mp.mpf(-new_new_logeps / 2))
        a_u = optimal_renyi_order(C_s, re_u, lam)
        red_sec_u = reduced_security(C_s, a_u, re_u, lam)
        if red_sec_u > max_red_sec_u:
            max_a_u = a_u
            max_red_sec_u = red_sec_u
        # if red_sec_u < 96:
        #     red_sec_u = 0
        # else:
        #     red_sec_u -= 96
        red_sec_u_data.append(red_sec_u)
    max_u, min_u, ave_u, var_u = round(np.max(red_sec_u_data), 20), round(np.min(red_sec_u_data), 20), round(np.average(red_sec_u_data), 20), round(np.var(red_sec_u_data), 20)
    print("for table 6: ", max_u, min_u, ave_u, var_u) 
    print("for table 7 or table 9: ", round(max_a_u, 2), round(max_red_sec_u, 2), "\n") 


# ----------------------------------------------- Figures ------------------------------------------
def gen_epsilon_i(n, logeps, CNT = 100):
    filename = f"B_data_{n}_ssh.npy"
    loaded = np.load(filename)
    r_data = np.zeros(n)
    if n == 512:
        sigma = 165.736617183
        the_logeps = 35.5
    elif n == 1024:
        sigma = 168.388571447
        the_logeps = 36

    sigma = sigma / smooth(the_logeps, 2 * n) * smooth(logeps, 2 * n)
    sum = 0
    for cnt in range (0, CNT):
        #r_data = gen_r_date(n)
        for j in range(n):
            r_data[j] = sigma / loaded[j][cnt]
        logeps_data = []
        for i in range(0, n):
            logeps_data.append(anti_smooth(r_data[i], 1)) 
        
        delta = 1
        for i in range(n):
            delta *= (1 + mp.power(2, -logeps_data[i])) / (1 - mp.power(2, -logeps_data[i])) 
        sum +=-log(delta ** 2 - 1, 2)

    return sum / CNT

# for i in range(20, 41, 1):
#     print(gen_epsilon_i(512, i))

# ----------------------------------------------- Table 4 & Table 5-----------------------------------

# print_falcon_statistics(512, 1000)
# print("\n")
# print_falcon_statistics(1024, 1000)

# comp_falcon_uf(512)
# print("\n")
# comp_falcon_uf(1024)

# ----------------------------------------------- Table 6 & Table 7-----------------------------------

# opt_new_ws_falcon_uf_suf(512, 2048, 21.5, 1, 1000) # 1 means uf
# opt_new_ws_falcon_uf_suf(512, 2048, 22.5, 1, 1000)
# opt_new_ws_falcon_uf_suf(512, 2048, 23.3, 1, 1000)
# # print("\n")
# opt_new_ws_falcon_uf_suf(1024, 4096, 21.5, 1, 1000)
# opt_new_ws_falcon_uf_suf(1024, 4096, 22.5, 1, 1000)
# opt_new_ws_falcon_uf_suf(1024, 4096, 23.5, 1, 1000) 

# ----------------------------------------------- Table 8 -----------------------------------

print_new_ws_falcon_uf_suf(512, 2048, 23.3, 1) # 1 means uf
print("\n")
print_new_ws_falcon_uf_suf(1024, 4096, 21.5, 1)

# ----------------------------------------------- Table 9 -----------------------------------

# comp_falcon_suf(512)
# comp_falcon_suf(1024)

# ----------------------------------------------- Table 10 -----------------------------------

# opt_new_ws_falcon_uf_suf(512, 12289, 20.5, sqrt(2), 1000)
# opt_new_ws_falcon_uf_suf(1024, 12289, 20.5, sqrt(2), 1000)
# print_new_ws_falcon_uf_suf(512, 12289, 20.5, sqrt(2))
# print("\n")
# print_new_ws_falcon_uf_suf(1024, 12289, 20.5, sqrt(2))

