from statistics import median
from math import ceil, sqrt

def median_confidence(data, q = 0.5, z = 1.96):
    data.sort()
    n = len(data)
    j = ceil(n*q - z*sqrt(n*q*(1-q)))
    k = ceil(n*q + z*sqrt(n*q*(1-q)))
    med_val = median(data)
    low_bnd, up_bnd = data[j], data[k]
    confidence = max(abs(med_val - low_bnd), abs(med_val - up_bnd))
    return med_val, confidence
