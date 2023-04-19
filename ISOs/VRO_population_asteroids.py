import os
import numpy as np
from synthetic_population import synthetic_population, total_number
from utils import true2ecc, ecc2mean, mean2tp, absolute_magnitude_asteroid, moid, max_hc_distance_asteroid, year2sec
from tqdm import tqdm

au = 149597870700.0
mu = 1.32712440042e20  # standard gravitational parameter of the Sun

maximum_array_size = int(1e5)

time_of_simulation = 1  # years
n0 = 0.1  # number-density (for D > D_ref)
d_reff = 100 # meters
v_min = 1e3
v_max = 1e5
d = [10, 1000]
# alpha = [[-2], [-2.5], [-3]]
Alpha = [[-2]]

V_cut = 24.5

albedo=1

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
   
    for alpha in Alpha:  # SFD

        output_file = stars + '_' + str(min(d)) + '_' + str(max(d)) + '_' + str(alpha[0]) + '_' + str(albedo) + '_asteroids.ssm'

        output_file = 'staro.ssm'

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


        """
        Maximum heliocentric distance where the largest object from the population can be observed
        # The maximum diameter is set to 1 km
        """
        hc_max = max_hc_distance_asteroid(max(d), albedo, V_cut)  # This is OK

        """
        Since the OIF simulation lasts some predefined time (1 year in our case), objects which are initially
        further away from the Sun than hc_max may reach this heliocentric distance during the simulation time. 
        This means we need to increase the model sphere a bit. This increment is how much the fastest object 
        from the population can travel during that time, or 1 year * v_max 
        
        This gives the radius of our model sphere rm = hc_max + 1 year * v_max.
        
        Object inside this sphere might be at heliocentric distance where they are observable or
        they might be further away but can reach the observable heliocentric distance during the simulation time.
        Objects outside this are for sure not observable during the simulation time. 
        
        If we increase simulation time (e.g. to 10 years), then object which are initially very far away from the Sun
        can maybe reach observable heliocentric distance during that time. This increases the model sphere, number of 
        objects computational resources....
        """
        rm = hc_max + year2sec(time_of_simulation) * v_max / au  # This is OK

        ISO_total_number = total_number(rm=rm, n0=n0, v_min=v_min, v_max=v_max,
                                        u_Sun=u_Sun, v_Sun=v_Sun, w_Sun=w_Sun,
                                        sigma_vx=sigma_vx, sigma_vy=sigma_vy, sigma_vz=sigma_vz,
                                        vd=vd, va=va, R_reff=R_reff,
                                        speed_resolution=100, angle_resolution=90, dr=0.1,
                                        d_ref=d_reff, d=d, alpha=alpha) # This is OK

        number_of_runs = 1
        if ISO_total_number > maximum_array_size:
            number_of_runs = int(np.ceil(ISO_total_number / maximum_array_size))

        for run in range(number_of_runs):

            np.savetxt('progress_asteroids.txt', ['Run number {} out of {} for SFD={}, population {}.'.format(
                run + 1, number_of_runs,
                alpha[0], population)], fmt='%s')

            print('\n ------------------------------ \n Run number {} out of {} for albedo={},  SFD={}, population {}.'.format(
                run + 1, number_of_runs,
                albedo, alpha[0], population))

            q, e, f, inc, Omega, omega, D = synthetic_population(rm=rm, n0=n0 / number_of_runs, v_min=v_min,
                                                                 v_max=v_max,
                                                                 u_Sun=u_Sun, v_Sun=v_Sun, w_Sun=w_Sun,
                                                                 sigma_vx=sigma_vx, sigma_vy=sigma_vy,
                                                                 sigma_vz=sigma_vz,
                                                                 vd=vd, va=va, R_reff=R_reff,
                                                                 speed_resolution=100, angle_resolution=90, dr=0.1,
                                                                 d_ref=100, d=d, alpha=alpha) # This is OK
            # =============================================================================
            # Filtering out objects which certainly cannot be detected because their perihelion distance is greater then the minimum helocentric distance
            # where they can be observed
            # =============================================================================

            hc_max1 = np.zeros(len(D))

            '''
            For every object (given its diameter and albedo) we calculate maximum heliocentric distance 
            where the object can be observed
            '''
            for i in range(len(D)):
                hc_max1[i] = max_hc_distance_asteroid(D[i], albedo, V_cut)  # This is OK

            """
            SELECTION No. 1
            We select only object whose perihelion distance is smaller than hc_max1. If it is larger, that object cannot reach 
            heliocentric distance where it can be observed
            """
            selection1 = q < hc_max1 # This is OK

            e1 = e[selection1]
            f1 = f[selection1]
            inc1 = inc[selection1]
            Omega1 = Omega[selection1]
            omega1 = omega[selection1]
            D1 = D[selection1]
            q1 = q[selection1]
            hc_max2 = hc_max1[selection1]

            '''
            SELECTION No. 2
            Now we check if the object is inside observable time at initial moment
            Equation of hyperbolic orbit:
            r = a*(1-e*cosh(E)), where E is hyperbolic anomaly
            
            From this equation, given eccentricity, semi-major axis and maximum observable heliocentric distance
            we can calculate critical hyperbolic anomaly (when ISO is exactly at hc_max2) 
            '''
            Ecr = np.arccosh(1 / e1 - hc_max2 / e1 / (q1 / (1 - e1)))  # This is OK

            """
            corresponding critical mean anomaly (maximum where an object of a given size can be observed)
            from hyperbolic Kepler equation
            M = e *sinh(E) - E
            """

            M_max = e1 * np.sinh(Ecr) - Ecr  # This is OK

            """
            We calculate E and M for every object from the population
            """



            M = np.zeros(len(q1))  # current mean anomaly
            for i in range(len(q1)):
                E = true2ecc(f1[i], e1[i])  # This is OK
                M[i] = ecc2mean(E, e1[i])  # This is OK

            """
            We calculate mean motion
            """
            n = np.sqrt(mu / (np.abs(q1 / (1 - e1)) * au) ** 3)  # mean motion (This is OK)

            """
            Finally, we calculate minimum mean anomaly from which an object can reach hc_max during the simulation time
            """
            M_min = -M_max - n * year2sec(time_of_simulation) # This is OK


            """
            Final selection:
            
            If object is outside observable sphere and it is on outgoing branch or orbit (Mean anomaly > 0) it is excluded
            because it is surely non observable. For objects with positive mean anomaly (those which are on the outgoing branch)
            we take only those which are inside their observable spheres (which depands on D and albedo)
            
            
            For objects on incoming branches we take those which are currently observable but also those whose mean anomaly 
            is larger than M_min. This means that those object will reach their observable sphere during the OIF simualtion time. 
            """

            selection2 = np.logical_and(M > M_min, M < M_max) # This is OK
            e2 = e1[selection2]
            f2 = f1[selection2]
            inc2 = inc1[selection2]
            Omega2 = Omega1[selection2]
            omega2 = omega1[selection2]
            D2 = D1[selection2]
            q2 = q1[selection2]

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
            tp = mean2tp(ecc2mean(true2ecc(f_out[i], e_out[i]), e_out[i]), q_out[i] / (1 - e_out[i]), 59853.01679398148)
            H = absolute_magnitude_asteroid(D_out[i], albedo)
            iso_moid = moid(1.99330267, -0.1965350, 0, 0.01671022, 1., omega_out[i], Omega_out[i], inc_out[i], e_out[i],
                            q_out[i] / (1 - e_out[i]), np.deg2rad(1.))

            # writing file for OIF simulation
            np.savetxt(file, np.column_stack(
                [ISO_name, 'COM', np.round(q_out[i], 6), np.round(e_out[i], 6), np.round(np.rad2deg(inc_out[i]), 3),
                 np.round(np.rad2deg(Omega_out[i])), np.round(np.rad2deg(omega_out[i])), np.round(tp, 6),
                 np.round(H, 3),
                 59853.0, 1, 6, np.round(iso_moid[0], 6), 'MOPS']),
                       fmt='%s')

        file.close()
