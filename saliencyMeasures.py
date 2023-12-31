#! /usr/bin/env python3
'''
Implemented by Erwan DAVID (IPI, LS2N, Nantes, France), 2018

E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (...). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
'''

from functools import partial
import numpy as np
from numpy import random
from skimage import exposure
from skimage import img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt

import re, os, glob

EPSILON = np.finfo('float').eps

#### METRICS --
'''
Commonly used metrics for evaluating saliency map performance.

Most metrics are ported from Matlab implementation provided by http://saliency.mit.edu/
Bylinskii, Z., Judd, T., Durand, F., Oliva, A., & Torralba, A. (n.d.). MIT Saliency Benchmark.

Python implementation: Chencan Qian, Sep 2014

[1] Bylinskii, Z., Judd, T., Durand, F., Oliva, A., & Torralba, A. (n.d.). MIT Saliency Benchmark.
[repo] https://github.com/herrlich10/saliency
'''
def normalize(x, method='standard', axis=None):

	x = np.array(x, copy=False)

	if axis is not None:

		y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
		shape = np.ones(len(x.shape))
		shape[axis] = x.shape[axis]

		if method == 'standard':
			res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
		elif method == 'range':
			res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
		elif method == 'sum':
			res = x / np.float_(np.sum(y, axis=1).reshape(shape))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	else:

		if method == 'standard':
			res = (x - np.mean(x)) / np.std(x)
		elif method == 'range':
			res = (x - np.min(x)) / (np.max(x) - np.min(x))
		elif method == 'sum':
			res = x / float(np.sum(x))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')

	return res

def match_hist(image, cdf, bin_centers, nbins=256):
	image = img_as_float(image)
	old_cdf, old_bin = exposure.cumulative_distribution(image, nbins) # Unlike [1], we didn't add small positive number to the histogram
	new_bin = np.interp(old_cdf, cdf, bin_centers)
	out = np.interp(image.ravel(), old_bin, new_bin)
	return out.reshape(image.shape)

def KLD(p, q):
	p = normalize(p, method='sum')
	q = normalize(q, method='sum')
	return np.sum(np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0))

def AUC_Judd(saliency_map, fixation_map, jitter=False):
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
	# Jitter the saliency map slightly to disrupt ties of the same saliency value
	if jitter:
		saliency_map += random.rand(*saliency_map.shape) * 1e-7
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# Calculate AUC
	thresholds = sorted(S_fix, reverse=True)
	tp = np.zeros(len(thresholds)+2)
	fp = np.zeros(len(thresholds)+2)
	tp[0] = 0; tp[-1] = 1
	fp[0] = 0; fp[-1] = 1
	for k, thresh in enumerate(thresholds):
		above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
		tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
		fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
	return np.trapz(tp, fp) # y, x

def AUC_shuffled(saliency_map, fixation_map, other_map, n_rep=100, step_size=0.1):

    s_map = np.array(saliency_map, copy=True)
    f_map = np.array(fixation_map, copy=True) > 0.5
    o_map = np.array(other_map, copy=True) > 0.5
    if other_map.shape != fixation_map.shape:
        raise ValueError('other_map.shape != fixation_map.shape')
    if not np.any(f_map):
        print('no fixation to predict')
        return np.nan
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape, order=3, mode='nearest')
    s_map = normalize(s_map, method='range')

    S = s_map.ravel()
    F = f_map.ravel()
    Oth = o_map.ravel()

    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)

    ind = np.nonzero(Oth)[0]
    n_ind = len(ind)
    n_fix_oth = min(n_fix,n_ind)

    r = random.randint(0, n_ind, [n_ind, n_rep])[:n_fix_oth,:]
    S_rand = S[ind[r]]

    auc = np.zeros(n_rep) * np.nan

    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix_oth)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)

def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# For each fixation, sample n_rep values from anywhere on the saliency map
	if rand_sampler is None:
		r = random.randint(0, n_pixels, [n_fix, n_rep])
		S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
	else:
		S_rand = rand_sampler(S, F, n_rep, n_fix)
	# Calculate AUC per random split (set of random locations)
	auc = np.zeros(n_rep) * np.nan
	for rep in range(n_rep):
		thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
		tp = np.zeros(len(thresholds)+2)
		fp = np.zeros(len(thresholds)+2)
		tp[0] = 0; tp[-1] = 1
		fp[0] = 0; fp[-1] = 1
		for k, thresh in enumerate(thresholds):
			tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
			fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
		auc[rep] = np.trapz(tp, fp)
	return np.mean(auc) # Average across random splits

def NSS(saliency_map, fixation_map):
	s_map = np.array(saliency_map, copy=False)
	f_map = np.array(fixation_map, copy=False) > 0.5
	if s_map.shape != f_map.shape:
		s_map = resize(s_map, f_map.shape)
	# Normalize saliency map to have zero mean and unit std
	s_map = normalize(s_map, method='standard')
	# Mean saliency value at fixation locations
	return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)

	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have zero mean and unit std

	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')
	# Compute correlation coefficient
	return np.corrcoef(map1.ravel(), map2.ravel())[0,1]


def SIM(saliency_map1, saliency_map2):
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have values between [0,1] and sum up to 1
	map1 = normalize(map1, method='range')
	map2 = normalize(map2, method='range')
	map1 = normalize(map1, method='sum')
	map2 = normalize(map2, method='sum')
	# Compute histogram intersection
	intersection = np.minimum(map1, map2)
	return np.sum(intersection)
#### METRICS --

# Name: func, symmetric?, second map should be saliency or fixation?
metrics = {
	"AUC_Judd": [AUC_Judd, False, 'fix'], # Binary fixation map
	"AUC_Borji": [AUC_Borji, False, 'fix'], # Binary fixation map
	"NSS": [NSS, False, 'fix'], # Binary fixation map
	"CC": [CC, False, 'sal'], # Saliency map
	"SIM": [SIM, False, 'sal'], # Saliency map
	"KLD": [KLD, False, 'sal'] } # Saliency map

# Possible float precision of bin files
dtypes = {16: np.float16,
		  32: np.float32,
		  64: np.float64}

get_binsalmap_infoRE = re.compile("(\d+_\w+)_(\d+)x(\d+)x(\d+)_(\d+)b")

def get_binsalmap_info(filename):

	name, width, height, Nframes, dtype = get_binsalmap_infoRE.findall(filename.split(os.sep)[-1])[0]
	width, height, Nframes, dtype = int(width), int(height), int(Nframes), int(dtype)

	return name, width, height, Nframes

def getSimVal(salmap1, salmap2, fixmap1=None, fixmap2=None):

	values = []

	for metric in keys_order:

		func = metrics[metric][0]
		sim = metrics[metric][1]
		compType = metrics[metric][2]

		if not sim:
			if compType == "fix" and not "NoneType" in [type(fixmap1), type(fixmap2)]:
				m = (func(salmap1, fixmap2)\
				   + func(salmap2, fixmap1))/2
			else:
				m = (func(salmap1, salmap2)\
				   + func(salmap2, salmap1))/2
		else:
			m = func(salmap1, salmap2)

		values.append(m)

	return values

def getFramePoolingIdx(tempWindowSize, FrameCount):

	tempWindowSize = int( np.round(FrameCount / 20 * (tempWindowSize/1000)))
	frameLost = FrameCount % tempWindowSize

	framePooling = np.arange(0, FrameCount+1, tempWindowSize) + frameLost//2
	framePooling = np.concatenate([ framePooling[:-1, None],
									framePooling[ 1:, None]-1 ], axis=1).astype(int)
	return framePooling

def getPooledFramesSM(file, range_, shape, dtype=32):
	iStart , iEnd = range_
	width, height = shape
	N = iEnd-iStart +1

	file.seek(width*height * iStart * (dtype//8))

	data = np.fromfile(file, count=width*height*N, dtype=dtypes[dtype])
	data = data.reshape([N, height, width])

	salmap = data.sum(axis=0)
	# plt.imshow(salmap);plt.show();exit()

	# Return saliency maps normalized
	return salmap / salmap.sum()

def uniformSphereSampling(N):

	gr = (1 + np.sqrt(5))/2
	ga = 2 * np.pi * (1 - 1/gr)

	ix = iy = np.arange(N)

	lat = np.arccos(1 - 2*ix/(N-1))
	lon = iy * ga

	lon %= 2*np.pi

	return np.concatenate([lat[:, None], lon[:, None]], axis=1)

def getPooledFramesFM(fixations, range_, shape):
	iStart, iEnd = range_

	ii = 0
	fixationmap = np.zeros(shape, dtype=int)
	for iFrame in range(iStart, iEnd+1):
		FIX = np.where(
				np.logical_and( fixations[:, 2] <= iFrame,
								fixations[:, 3] >= iFrame ) )[0]
		for iFix in FIX:
			fixationmap[ int(round(fixations[iFix, 1])), int(round(fixations[iFix, 0])) ] += 1
			ii+=1
	# plt.imshow(np.fliplr(fixationmap));plt.show();exit()
	return np.fliplr(fixationmap)

if __name__ == "__main__":
	from time import time
	t1 = time()
	# Similarité metrics to compute and output to file
	# keys_order = ['AUC_Judd', 'AUC_Borji', 'NSS', 'CC', 'SIM', 'KLD']
	keys_order = ['AUC_Borji', 'NSS', 'CC', 'SIM', 'KLD']

	SM_PATH = "../HE/SalMaps/"
	SP_PATH = "../HE/Scanpaths/R/"

	"""
	One issue when comparing saliency maps in equirectangular format, is that poles of the sphere are over-represented because of the latitudinal distortions
	One possible correction is to take N points uniformely sampled on a sphere
		see blog.wolfram.com/2011/07/28/how-i-made-wine-glasses-from-sunflowers/
		A bigger N means a better approximation, the solution given above shows irregularities at sin = 0
	We know an equirectangular map's vetical distortion is a function of sin(y/np.pi). We propose as a second solution to multiply all rows of the saliency matrix with a weight vector modeled:
		sin(linspace(0, pi, height)) - O to pi with as many steps as vertical pixels
	"""
	SAMPLING_TYPE = [ # Different sampling method to apply to saliency maps
		"Sphere_9999999", # Too many points
		"Sphere_1256637", # 100,000 points per steradian
		"Sphere_10000",   # 10,000
		"Sin",			  # Sin(height)
		"Equi"			  # None
		]

	SAMPLING_TYPE = SAMPLING_TYPE[-2] # Sin weighting by default
	print("SAMPLING_TYPE: ", SAMPLING_TYPE)

	# Path to vieo saliency maps we wish to compare
	salmap1_path = SM_PATH + "1_PortoRiverside_2048x1024x500_32b.bin"
	salmap2_path = SM_PATH + "2_Diner_2048x1024x600_32b.bin"

	salmap1_file = open(salmap1_path, "rb")
	salmap2_file = open(salmap2_path, "rb")

	scanpath1_path = SP_PATH + "1_PortoRiverside_fixations.csv"
	scanpath2_path = SP_PATH + "2_Diner_fixations.csv"

	name1, width, height, Nframes = get_binsalmap_info(salmap1_path)
	name2, _, _, _ = get_binsalmap_info(salmap2_path)

	if name1 != name2:
		print("Warning: you are comparing saliency maps from different stimuli. They may not have the same number of frames.")

	if SAMPLING_TYPE.split("_")[0] == "Sphere":

		print(int(SAMPLING_TYPE.split("_")[1]))
		unifS = uniformSphereSampling( int(SAMPLING_TYPE.split("_")[1]))

		unifS[:, 0] = unifS[:, 0] / np.pi * (height-1)
		unifS[:, 1] = unifS[:, 1] / (2*np.pi) * (width-1)

		unifS = unifS.astype(int)

	elif SAMPLING_TYPE == "Sin":
		VerticalWeighting = np.sin(np.linspace(0, np.pi, height)) # latitude weighting
		# plt.plot(np.arange(height), VerticalWeighting);plt.show()

	# Load fixation lists
	fixations1 = np.loadtxt(scanpath1_path, delimiter=",", skiprows=1, usecols=(1,2, 5,6))
	fixations2 = np.loadtxt(scanpath2_path, delimiter=",", skiprows=1, usecols=(1,2, 5,6))

	fixations1 = fixations1 * [width, height, 1,1] - [1,1, 0,0]
	fixations2 = fixations2 * [width, height, 1,1] - [1,1, 0,0]

	# Get start/end of frames given a time window
	fPool = getFramePoolingIdx(200, Nframes)

	with open("example_SalmapComparisons.csv", "w") as saveFile:
		saveFile.write("stimName, iPoolFrame, metric, value\n")

		for iFrame in range(fPool.shape[0]):
				print(" "*20, "\r{}/{} - {:.2f}% - ".format(iFrame+1, fPool.shape[0], (iFrame+1)/fPool.shape[0]*100), end="")

				# Retrieve saliency map for frame range
				salmap1_frame = getPooledFramesSM(salmap1_file, fPool[iFrame, :], [width, height])
				salmap2_frame = getPooledFramesSM(salmap2_file, fPool[iFrame, :], [width, height])

				# Retrieve fixations map for frame range
				fixmap1_frame = getPooledFramesFM(fixations1, fPool[iFrame, :], [height, width])
				fixmap2_frame = getPooledFramesFM(fixations2, fPool[iFrame, :], [height, width])

				# Apply uniform sphere sampling if specified
				if SAMPLING_TYPE.split("_")[0] == "Sphere":
					salmap1_frame = salmap1_frame[unifS[:, 0], unifS[:, 1]]
					salmap2_frame = salmap2_frame[unifS[:, 0], unifS[:, 1]]

					fixmap1_frame = fixmap1_frame[unifS[:, 0], unifS[:, 1]]
					fixmap2_frame = fixmap2_frame[unifS[:, 0], unifS[:, 1]]
				# Weight saliency maps vertically if specified
				elif SAMPLING_TYPE == "Sin":
					salmap1_frame = salmap1_frame * VerticalWeighting[:, None] + EPSILON
					salmap2_frame = salmap2_frame * VerticalWeighting[:, None] + EPSILON

				salmap1_frame = normalize(salmap1_frame, method='sum')
				salmap2_frame = normalize(salmap2_frame, method='sum')

				# Compute similarity metrics
				values = getSimVal(salmap1_frame, salmap2_frame,
								   fixmap1_frame, fixmap2_frame)
				# Log results
				for iVal, val in enumerate(values):
					saveFile.write("{}, {}, {}, {}\n".format(name1, iFrame, keys_order[iVal], val))

	salmap1_file.close()
	salmap2_file.close()
	print("")

	print("T_delta = {}".format(time() - t1))