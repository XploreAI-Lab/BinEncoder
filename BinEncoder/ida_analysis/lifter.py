
import argparse
import sys
import os
import subprocess
import json
import pandas as pd
from os.path import basename, join
from multiprocessing import Pool

from tqdm import tqdm


GSAT_BIN_PATH = r"C:\Users\tianh\Desktop\gsat-1.0.jar"
output_fp = r"E:\BinEncoder\dbs\Dataset-1\training"

def extract_one_pcode(bin_fp, cfg_summary_fp, output_dir):
    cmd = f"java -ea -Xmx16G -jar {GSAT_BIN_PATH} pcode-extractor-v2 -m elf \
        -f {bin_fp} -c {cfg_summary_fp} -of ALL -v 1\
        -opt 1 -o {output_dir}"
    proc = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE)
    out, _ = proc.communicate()
    code = proc.returncode

def find_matching_files(bin_dir, cfg_summary_dir):
    # 获取两个目录下的所有文件
    bin_files = os.listdir(bin_dir)
    cfg_summary_files = os.listdir(cfg_summary_dir)

    # 创建一个字典来存储文件名前缀和对应的文件路径
    bin_dict = {}
    cfg_summary_dict = {}

    # 遍历bin目录下的文件
    for file in bin_files:
        prefix = ''.join(file.split('.')[:-1])  # 获取第一个点左边的部分
        bin_dict[prefix] = os.path.join(bin_dir, file)
    # 遍历cfg_summary目录下的文件
    for file in cfg_summary_files:
        prefix = ''.join(file.split('.')[:-2])  # 获取第一个点左边的部分
        cfg_summary_dict[prefix] = os.path.join(cfg_summary_dir, file)

    # 找到匹配的文件对
    matching_pairs = []
    for prefix in bin_dict:
        if prefix in cfg_summary_dict:
            matching_pairs.append((bin_dict[prefix], cfg_summary_dict[prefix]))

    return matching_pairs

def process_matching_files(bin_dir, cfg_summary_dir):
    matching_pairs = find_matching_files(bin_dir, cfg_summary_dir)
    for bin_fp, cfg_summary_fp in tqdm(matching_pairs, desc="Processing files", unit="file"):
        output_dir = os.path.join(output_fp, basename(cfg_summary_fp))
        extract_one_pcode(bin_fp, cfg_summary_fp, output_dir)

if __name__ == '__main__':
    bin_dir = r"E:\BinEncoder\ida_analysis\ori_bin_files\ultimate-pretrain-binaries"
    summary_dir = r"E:\BinEncoder\dbs\Dataset-1\cfg_summary"
    process_matching_files(bin_dir, summary_dir)
