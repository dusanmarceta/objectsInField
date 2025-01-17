import os
import numpy as np
import sys
import os.path
import time



alpha = np.array(['-1.0', '-1.5', '-2.0', '-2.5', '-3.0', '-3.5', '-4.0'])


Alpha = []
for i in range(len(alpha)):
    Alpha.append([alpha[i]])

for population in range(3):  # for 3 populations

    if population == 0:
        stars = 'OB'
    elif population == 1:
        stars = 'M'
    elif population == 2:
        stars = 'G'

    for alpha in Alpha:  # SFD
        
        print('Population {}, alpha = {}'.format(stars, alpha))
        np.savetxt('progress_OIF.txt', ['Population {}, alpha = {}'.format(stars, alpha)], fmt='%s')

        
        input_file = stars + '_' + alpha[0] + '_ecl.ssm'
        output_file = stars + '_' + alpha[0] + '_ecl_OIF.txt'

        # # Generatong configure file for OIF
        q=np.loadtxt(input_file, usecols=2, skiprows=1)
        number_of_objects=str(len(q))

        with open('input.config', 'r') as file:
            data = file.readlines()

        file.close()
        data[1] = "Population model    = " + input_file + '\n'
        data[7] = "nObjects            = " + number_of_objects + '\n'

        with open('input.config', 'w') as file:
            file.writelines(data)

        file.close()
        # sys.exit()
        # Running OIF
        command = 'oif -f input.config > {}'.format(output_file)
        os.system(command)

        # wait until the simulation is finished
        while not os.path.exists(output_file):
            time.sleep(60)






