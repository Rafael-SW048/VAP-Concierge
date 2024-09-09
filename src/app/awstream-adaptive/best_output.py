import pandas as pd
import numpy as np
import os
from dds_utils import get_best_configuration


test = get_best_configuration(100, f"./data-set/india/profile/profile-90.csv")

print(test)