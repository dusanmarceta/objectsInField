import os
import numpy as np
from synthetic_population import synthetic_population, total_number
from utils import true2ecc, ecc2mean, mean2tp, absolute_magnitude, moid, max_hc_distance
from tqdm import tqdm

au = 149597870700.0

maximum_array_size = int(1e5)

time_of_simulation = 1.  # years
v_min = 1e3
v_max = 1e5
d = [50, 1000]
alpha = [[-2], [-2.5], [-3]]
albedo = 0.07
m_cut = 24.5
n0 = 0.1

u_Sun = 1e4
v_Sun = 1.1e4
w_Sun = 7e3

sigma_vx = 1.2e4
sigma_vy = 1.1e4
sigma_vz = 0.9e4
vd = np.deg2rad(36)
stars = 'OB'

# sigma_vx=3.1e4
# sigma_vy=2.3e4
# sigma_vz=1.6e4
# vd=np.deg2rad(7)
# stars='M'

# sigma_vx=2.6e4
# sigma_vy=1.8e4
# sigma_vz=1.5e4
# vd=np.deg2rad(12)
# stars='G'


va = 0
R_reff = 696340000.

# =============================================================================
#                               SIMULATION
# =============================================================================
for alpha in alpha:  # SFD

    output_file = stars + '_' + str(min(d)) + '_' + str(max(d)) + '_' + str(alpha[0]) + '.ssm'

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

    hc_max = max_hc_distance(max(d), albedo, m_cut)

    rm = hc_max + time_of_simulation * 356.25 * 86400 * v_max / au

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
        print('\n ------------------------------ \n Run number {} out of {} for SFD={}'.format(run + 1, number_of_runs,
                                                                                               alpha[0]))

        q, e, f, inc, Omega, omega, D = synthetic_population(rm=rm, n0=n0 / number_of_runs, v_min=v_min, v_max=v_max,
                                                             u_Sun=u_Sun, v_Sun=v_Sun, w_Sun=w_Sun,
                                                             sigma_vx=sigma_vx, sigma_vy=sigma_vy, sigma_vz=sigma_vz,
                                                             vd=vd, va=va, R_reff=R_reff,
                                                             speed_resolution=100, angle_resolution=90, dr=0.1,
                                                             d_ref=1000, d=d, alpha=alpha)
        # =============================================================================
        # Filtering out objects which certainly cannot be detected because their perihelion distance is greater then the minimum helocentric distance
        # where they can be observed
        # =============================================================================

        hc_max = max_hc_distance(D, albedo, m_cut)

        q_out = np.append(q_out, q[q < hc_max])
        e_out = np.append(e_out, e[q < hc_max])
        f_out = np.append(f_out, f[q < hc_max])
        inc_out = np.append(inc_out, inc[q < hc_max])
        Omega_out = np.append(Omega_out, Omega[q < hc_max])
        omega_out = np.append(omega_out, omega[q < hc_max])
        D_out = np.append(D_out, D[q < hc_max])

    # Making file for OIF simulation
    # opening output file 
    file = open(output_file, 'ab')
    # first line
    np.savetxt(file, ['!!OID FORMAT q e i node argperi t_p H t_0 INDEX N_PAR MOID COMPCODE'], fmt='%s')

    for i in tqdm(range(len(q_out))):
        ISO_name = 'ISO_' + str(i)
        tp = mean2tp(ecc2mean(true2ecc(f_out[i], e_out[i]), e_out[i]), q_out[i] / (1 - e_out[i]), 59200.0)
        H = absolute_magnitude(D_out[i], albedo)
        iso_moid = moid(1.99330267, -0.1965350, 0, 0.01671022, 1., omega_out[i], Omega_out[i], inc_out[i], e_out[i],
                        q_out[i] / (1 - e_out[i]), np.deg2rad(1.))

        # writing file for VRO simulation
        np.savetxt(file, np.column_stack(
            [ISO_name, 'COM', np.round(q_out[i], 6), np.round(e_out[i], 6), np.round(np.rad2deg(inc_out[i]), 3),
             np.round(np.rad2deg(Omega_out[i])), np.round(np.rad2deg(omega_out[i])), np.round(tp, 6), np.round(H, 3),
             54466.0, 1, 6, np.round(iso_moid[0], 6), 'MOPS']),
                   fmt='%s')

    file.close()
