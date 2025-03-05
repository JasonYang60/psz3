import subprocess
import os
import numpy as np
import shutil

def copy_file(src, dst):
    shutil.copy(src, dst)


def run_cpp_and_append_output(executable_path, args, out_path):

    cmd = [executable_path] + args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        output = result.stdout
        print(result.stderr)
        
        with open(out_path, 'a', encoding='utf-8') as f:
            # f.write(output)
            f.write(result.stderr)
        
        print(f"output appended to {out_path}")
    
    except subprocess.CalledProcessError as e:
        print(f"error:\n{e.stderr}")

def calcPSNR(ori, cmp):
    diff = ori - cmp

    rg = np.max(ori) - np.min(ori)
    mse = np.mean(diff ** 2)
    psnr = 10 * np.log10((rg**2)/mse)
    return psnr

def difference_of_two_files(file1, file2, count, dtype='<f4'):
    """
    """
    arr1 = np.fromfile(file1, dtype=dtype, count=count)
    arr2 = np.fromfile(file2, dtype=dtype, count=count)
    
    diff = arr1 - arr2

    return diff

def delta_zfp(exePath, datasetName, dim, eb_base, eb_list, outPath, dtype='<f4'):
    datasetCopy = datasetName + '.zfp.temp'
    datasetPath = '/home/zyang/Desktop/datasets/'

    copy_file(datasetPath + datasetName, datasetPath + datasetCopy)

    iFile = datasetPath + datasetCopy
    oFile = iFile + '.out'

    num = 1
    for d in dim:
        num = d * num


    
    ori = np.fromfile(iFile, dtype=dtype, count=num)
    with open(outPath, "w", encoding="utf-8") as f:
        f.write("") 

    arr = np.zeros(num)
    for i in eb_list:

        dim_list = ['-' + str(len(dim))]
        for d in dim:
            dim_list.append(str(d))
        
        # args = ['-i', iFile, '-o', oFile, '-s', '-d'] + dim_list + ['-a', str(eb_base * i)]
        args = ['-i', iFile, '-o', oFile, '-s', '-d'] + dim_list + ['-r', str(eb_base * i)]
        run_cpp_and_append_output(exePath, args, outPath)

        # arr = arr + np.fromfile(oFile, dtype=dtype, count=num)

        # diff = difference_of_two_files(iFile, oFile, num, dtype=dtype)
    
        # diff.astype(dtype).tofile(iFile)
        # with open(outPath, 'a', encoding='utf-8') as f:
        #     f.write('PSNR = ' + str(calcPSNR(ori, arr)) + '\n\n')

if __name__ == "__main__":
#     DATA_NAME_LIST=("density.d64" "pressure.d64" "Uf48.f64.bin.dat" "sample_r_B_0.5_26.d64" "stat_planar.1.1000E-03.field.d64.1" "631-tst.bin.d64")
# DIMENSIONS_LIST=("-3 384 384 256" "-3 384 384 256" "-3 500 500 100" "-1 33554433" "-3 500 500 500" "-1 102953248")

    dataset_list = ['density.d64', 'pressure.d64', 'Uf48.f64.bin.dat', "sample_r_B_0.5_26.d64", "stat_planar.1.1000E-03.field.d64.1", "631-tst.bin.d64"]
    # eb_base_list = [2e-9, 4.40e-09, 9.26e-8, 4.18e-08, 3.92e-11, 5.24417e-09]
    # eb_list = [65536, 4096, 256, 16, 1]
    eb_base_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # eb_list = [1, 1, 1, 1, 1]
    eb_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    dim_list = [[384, 384, 256], [384, 384, 256], [500, 500, 100], [33554433], [500, 500, 500], [102953248]]
    dtype = '<f8'
    exePath = './compressors/zfp_delta/zfp'

    for datasetName, dim, eb_base in zip(dataset_list, dim_list, eb_base_list):

        outPath = './log/zfp_delta.log/' + datasetName + '.log'
        delta_zfp(exePath, datasetName, dim, eb_base, eb_list, outPath, dtype=dtype)

    # args[-1] = str(eb)
    # run_cpp_and_append_output(exePath, args, outPath)




