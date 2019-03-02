import subprocess
from sys import platform
#from itertools import permutations, repeat
import itertools
import glob
import os

path = '/share/Daten/Volumen/mehr/steeb/disk3_new/disk3/'
dry = '/share/Daten/Volumen/mehr/steeb/FB01_dry_highres/'
out_path = '/share/Daten/Volumen/mehr/steeb/disk3_new/disk3/conv2'

if not os.path.exists(out_path):
    os.makedirs(out_path)

def runCMD(cmd_in):

    cmd = []
    for c in cmd_in:
        if type(c) is str:
            cmd.append(c)
        else:
            cmd.append(str(c))
    
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)

    out, err = p.communicate()

    #print(err, out)

    if err:
        print('Error string ' + str(err))

    return str(out)


volumes_full = glob.glob(path + '*_co2_*')

#Volumes_full = glob.glob(path + '*_*')
#volumes_full.append(dry)
print(volumes_full)

volumes = []
for volume in volumes_full:
    if os.path.exists(volume + '/rec_DMP_0'):
        print(volume + ' is valid')
        volumes.append(volume)
    else:
        print(volume + ' is invalid')

        

conv_cmd = ['./conv_dmp2raw', out_path]
conv_cmd = conv_cmd + volumes

print(conv_cmd)
runCMD(conv_cmd)

configs = glob.glob(out_path + '/*.config')

for config in configs:
    down_cmd = ['../volData/downsample', '2', config[:-7] + '_r2.raw']
    print(down_cmd)
    runCMD(down_cmd)
