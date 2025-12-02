#!/bin/python3
CODING_CHALLENGE = "coding-challenge-templates/python3"

import argparse
import logging
from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate, tzip, tmap
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import *
import sys
sys.setrecursionlimit(99999)

# from collections import deque
# import copy
# from dataclasses import dataclass, field
# from functools import lru_cache
# import glob
# from natsort import natsorted
from math import *
# import multiprocessing
# from multiprocessing import Pool
# import numpy as np
# import os
from pdb import set_trace as st
from pprint import pp
# import random
from time import sleep
# import z3


DATA_DIR = 'io'

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())
LOG_HANDLER = logging.StreamHandler()
LOG_ENABLE = lambda: LOG.addHandler(LOG_HANDLER)
LOG_DISABLE = lambda: LOG.removeHandler(LOG_HANDLER)
logging.basicConfig(
  format="[%(asctime)s.%(msecs)03d %(levelname)8s] %(message)s   (%(funcName)s@%(filename)s:%(lineno)s)",
  datefmt="%H:%M:%S",
)
LOG.setLevel(logging.DEBUG)

CONFIG = None # will be generated via argparse


def process(filename: str) -> Any:
  with open(filename, 'r') as fin:
    with open(f'{".".join(filename.split(".")[:-1])}.out', 'w') as fout:
      # Process a few lines manually here before looping line-by-line
      # n = int(next(fout))
      for line in tqdm(fin, f'{filename} lines'):
        line = line.rstrip()

        # Process input file line-by-line here

        out = f"{line}"
        fout.write(f"{out}\n")
        LOG.info(out)
      return None

def test() -> None:
  tests = {
    # <function>: (test1, test2, ...)
    int: (
      # Add tests here
      (("4",), 4),
    ),
    # Add tests for other functions here
    lambda *args: list(range(*args)): (
      ((1,), [0]),
      ((1, 3), [1, 2]),
    ),
    nums: (
      (("1 2 3",), [1, 2, 3]),
      (("1 2 3", None, float), [1.0, 2.0, 3.0]),
      (("1, 2, 3", ', '), [1, 2, 3]),
      (("1-1, 2-2, 3-3", [',', '-']), [[1,1], [2,2], [3,3]]),
      (("1-1, 2-2, 3-3", [',', '-'], float), [[1.0,1.0], [2.0,2.0], [3.0,3.0]]),
    ),
  }
  for func, func_tests in tqdm(tests.items(), f"Testing"):
    for test in tqdm(func_tests):
      args, expected = test
      actual = func(*args)
      if actual != expected:
        LOG.error(f"test failed: {args} -> {actual} != {expected}")
        exit(1)

def main(input_files: list[str]) -> None:
  test()
  LOG.info("Tests completed successfully.")
  if CONFIG.tests_only:
    exit(0)
  try:
    # results = {}
    for f in tqdm(input_files, 'process input files'):
      LOG.critical(f"Processing {f}")
      res = process(f)
      # results[f] = res
      LOG.critical(f"Processed {f} -> {res}")
    #pp(results)
  except KeyboardInterrupt as e:
    LOG.error(str(e), exc_info=True)
    exit(130)


def parse_args():
  parser = argparse.ArgumentParser(
    description = CODING_CHALLENGE,
  )
  parser.add_argument('input_files', nargs='*', type=str)
  parser.add_argument('-t', '--tests-only', action='store_true')
  return parser.parse_args()
                         
#
#  ▄▄    ▄▄               ██     ▄▄▄▄               
#  ██    ██    ██         ▀▀     ▀▀██               
#  ██    ██  ███████    ████       ██      ▄▄█████▄ 
#  ██    ██    ██         ██       ██      ██▄▄▄▄ ▀ 
#  ██    ██    ██         ██       ██       ▀▀▀▀██▄ 
#  ▀██▄▄██▀    ██▄▄▄   ▄▄▄██▄▄▄    ██▄▄▄   █▄▄▄▄▄██ 
#    ▀▀▀▀       ▀▀▀▀   ▀▀▀▀▀▀▀▀     ▀▀▀▀    ▀▀▀▀▀▀  
#

def nums(numbers: str, separator=None, func=int):
  if isinstance(separator, list):
    if len(separator) == 1:
      return nums(numbers, separator[0], func=func)
    splits = numbers.split(separator[0])
    return [nums(split, separator[1:], func=func) for split in splits]
  else:
    return list(map(func, numbers.split(separator)))
def ints(*args, **kwargs): return nums(*args, func=int, **kwargs)
def floats(*args, **kwargs): return nums(*args, func=float, **kwargs)

################################################################################

if __name__ == '__main__':
  with logging_redirect_tqdm():
    CONFIG = parse_args()
    main(CONFIG.input_files)

#   _______           _______  _______ _________ _______           _______  _______ _________
#  (  ____ \|\     /|(  ____ \(  ___  )\__   __/(  ____ \|\     /|(  ____ \(  ____ \\__   __/
#  | (    \/| )   ( || (    \/| (   ) |   ) (   | (    \/| )   ( || (    \/| (    \/   ) (
#  | |      | (___) || (__    | (___) |   | |   | (_____ | (___) || (__    | (__       | |
#  | |      |  ___  ||  __)   |  ___  |   | |   (_____  )|  ___  ||  __)   |  __)      | |
#  | |      | (   ) || (      | (   ) |   | |         ) || (   ) || (      | (         | |
#  | (____/\| )   ( || (____/\| )   ( |   | |   /\____) || )   ( || (____/\| (____/\   | |
#  (_______/|/     \|(_______/|/     \|   )_(   \_______)|/     \|(_______/(_______/   )_(
#

r"""
dataclass
# https://docs.python.org/3/library/dataclasses.html
# https://realpython.com/python-data-classes/
@dataclass
class Entity:
  value_str: str
  value_int: int
  value_obj: field(default_factory=lambda: ["some custom object"])


@lru_cache
def count_vowels(sentence):
    return sum(sentence.count(vowel) for vowel in 'AEIOUaeiou')


# https://towardsdatascience.com/a-complete-guide-to-using-progress-bars-in-python-aa7f4130cda8
# https://github.com/tqdm/tqdm#parameters
tqdm(
  iterable=None,
  desc=None,
  total=None,
  leave=True,           # Keep progressbar after completion
  file=None,
  ncols=None,
  mininterval=0.1,
  maxinterval=10.0,
  miniters=None,
  ascii=None,
  disable=False,
  unit='it',
  unit_scale=False,
  dynamic_ncols=False,
  smoothing=0.3,
  bar_format=None,
  initial=0,
  position=None,        # level of nestedness [0,...)
  postfix=None,
  unit_divisor=1000)
with tqdm(...) as t:
  t.update(1)


# http://theory.stanford.edu/~nikolaj/programmingz3.html
# https://ericpony.github.io/z3py-tutorial/guide-examples.htm
s = z3.Solver()
s.check()
s.model()
"""
