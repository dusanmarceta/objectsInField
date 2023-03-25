import os
import numpy as np
from synthetic_population import synthetic_population, total_number
from utils import true2ecc, ecc2mean, mean2tp, absolute_magnitude_comet, moid, max_hc_distance_comet, year2sec, \
    max_hc_distance_asteroid

au = 149597870700.0
mu = 1.32712440042e20  # standard gravitational parameter of the Sun

maximum_array_size = int(1e5)

time_of_simulation = 1  # years
n0 = 0.1  # number-density (for D > 1 km)
v_min = 1e3
v_max = 1e5
d = [50, 1000]
alpha = [[-2], [-2.5], [-3]]

V_cut = 24.5

b1 = -0.2
b2 = 3.1236
n = 5  # brightening due to activity

u_Sun = 1e4
v_Sun = 1.1e4
w_Sun = 7e3

sigma = [[1.2e4, 3.1e4, 2.6e4], [1.2e4, 2.3e4, 1.8e4],
         [0.9e4, 1.6e4, 1.5e4]]  # velocity dispersions for 3 stellar populations

vertex_deviation = [np.deg2rad(36), np.deg2rad(7), np.deg2rad(12)]  # vertex deviation for 3 stellar populations
va = 0  # asymmetrical drift
R_reff = 696340000.  # radius of the Sun

for population in range(1):  # for 3 populations

    sigma_vx = sigma[0][population]
    sigma_vy = sigma[1][population]
    sigma_vz = sigma[2][population]
    vd = vertex_deviation[population]

    if population == 0:
        stars = 'OB'
    elif population == 1:
        stars = 'M'
    elif population == 2:
        stars = 'G'
    # =============================================================================
    #                               SIMULATION
    # =============================================================================
    for alpha in alpha:  # SFD

        output_file = stars + '_' + str(min(d)) + '_' + str(max(d)) + '_' + str(alpha[0]) + '_comets.ssm'

        q_out = np.array([])
        e_out = np.array([])
        f_out = np.array([])
        inc_out = np.array([])
        Omega_out = np.array([])
        omega_out = np.array([])
        D_out = np.array([])

        # Deleting the output file if already exists
        if os.path.exists(output_file):
            os.remove(output_file)

        hc_max = max_hc_distance_comet(max(d), b1, b2, n, V_cut)

        rm = hc_max + year2sec(time_of_simulation) * v_max / au

        ISO_total_number = total_number(rm=rm, n0=n0, v_min=v_min, v_max=v_max,
                                        u_Sun=u_Sun, v_Sun=v_Sun, w_Sun=w_Sun,
                                        sigma_vx=sigma_vx, sigma_vy=sigma_vy, sigma_vz=sigma_vz,
                                        vd=vd, va=va, R_reff=R_reff,
                                        speed_resolution=100, angle_resolution=90, dr=0.1,
                                        d_ref=1000, d=d, alpha=alpha)

        number_of_runs = 1
        if ISO_total_number > maximum_array_size:
            number_of_runs = int(np.ceil(ISO_total_number / maximum_array_size))

        for run in range(number_of_runs):
            np.savetxt('progress_comets.txt', ['Run number {} out of {} for SFD={}, population {}.'.format(
                run + 1, number_of_runs,
                alpha[0], population)], fmt='%s')
            print('\n ------------------------------ \n [Run number {} out of {} for SFD={}, population {}.'.format(
                run + 1, number_of_runs,
                alpha[0], population))

            q, e, f, inc, Omega, omega, D = synthetic_population(rm=rm, n0=n0 / number_of_runs, v_min=v_min,
                                                                 v_max=v_max,
                                                                 u_Sun=u_Sun, v_Sun=v_Sun, w_Sun=w_Sun,
                                                                 sigma_vx=sigma_vx, sigma_vy=sigma_vy,
                                                                 sigma_vz=sigma_vz,
                                                                 vd=vd, va=va, R_reff=R_reff,
                                                                 speed_resolution=100, angle_resolution=90, dr=0.1,
                                                                 d_ref=1000, d=d, alpha=alpha)
            # =============================================================================
            # Filtering out objects which certainly cannot be detected because their perihelion distance is greater then the minimum helocentric distance
            # where they can be observed
            # =============================================================================

            hc_max1 = np.zeros(len(D))

            for i in range(len(D)):
                hc_max_comet = max_hc_distance_comet(D[i], b1, b2, n, V_cut)
                hc_max_asteroid = max_hc_distance_asteroid(D[i], 1., V_cut)
                hc_max1[i] = np.max([hc_max_comet, hc_max_asteroid])

            selection = q < hc_max1
            e1 = e[selection]
            f1 = f[selection]
            inc1 = inc[selection]
            Omega1 = Omega[selection]
            omega1 = omega[selection]
            D1 = D[selection]
            q1 = q[selection]
            hc_max2 = hc_max1[selection]

            Ecr = np.arccosh(1 / e1 - hc_max2 / e1 / (q1 / (1 - e1)))  # OK

            # corresponding critical mean anomaly
            M_max = e1 * np.sinh(Ecr) - Ecr  # OK

            M = np.zeros(len(q1))  # current mean anomaly
            for i in range(len(q1)):
                E = true2ecc(f1[i], e1[i])  # OK
                M[i] = ecc2mean(E, e1[i])  # OK

            mm = np.sqrt(mu / (np.abs(q1 / (1 - e1)) * au) ** 3)  # mean motion
            M_min = -M_max - mm * year2sec(time_of_simulation)

            # Taking only those which are currently observable
            # or a bit behind the observable zone but can reach it during
            # simulation time

            selection = np.logical_and(M > M_min, M < M_max)
            e2 = e1[selection]
            f2 = f1[selection]
            inc2 = inc1[selection]
            Omega2 = Omega1[selection]
            omega2 = omega1[selection]
            D2 = D1[selection]
            q2 = q1[selection]

            q_out = np.append(q_out, q2)
            e_out = np.append(e_out, e2)
            f_out = np.append(f_out, f2)
            inc_out = np.append(inc_out, inc2)
            Omega_out = np.append(Omega_out, Omega2)
            omega_out = np.append(omega_out, omega2)
            D_out = np.append(D_out, D2)

        # Making file for OIF simulation
        # opening output file
        file = open(output_file, 'ab')
        # first line
        np.savetxt(file, ['!!OID FORMAT q e i node argperi t_p H t_0 INDEX N_PAR MOID COMPCODE'], fmt='%s')

        for i in range(len(q_out)):
            ISO_name = 'ISO_' + str(i)
            tp = mean2tp(ecc2mean(true2ecc(f_out[i], e_out[i]), e_out[i]), q_out[i] / (1 - e_out[i]), 59200.0)
            H = absolute_magnitude_comet(D_out[i], b1, b2)
            iso_moid = moid(1.99330267, -0.1965350, 0, 0.01671022, 1., omega_out[i], Omega_out[i], inc_out[i], e_out[i],
                            q_out[i] / (1 - e_out[i]), np.deg2rad(1.))

            # writing file for OIF simulation
            np.savetxt(file, np.column_stack(
                [ISO_name, 'COM', np.round(q_out[i], 6), np.round(e_out[i], 6), np.round(np.rad2deg(inc_out[i]), 3),
                 np.round(np.rad2deg(Omega_out[i])), np.round(np.rad2deg(omega_out[i])), np.round(tp, 6),
                 np.round(H, 3),
                 54466.0, 1, 6, np.round(iso_moid[0], 6), 'MOPS']),
                       fmt='%s')

        file.close()
