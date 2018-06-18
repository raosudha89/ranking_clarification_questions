import argparse
import numpy as np
import pdb
import random
from sklearn.metrics import average_precision_score
import sys

BAD_QUESTIONS="unix_56867 unix_136954 unix_160510 unix_138507".split() + \
			"askubuntu_791945 askubuntu_91332 askubuntu_704807 askubuntu_628216 askubuntu_688172 askubuntu_727993".split() + \
			"askubuntu_279488 askubuntu_624918 askubuntu_527314 askubuntu_182249 askubuntu_610081 askubuntu_613851 askubuntu_777774 askubuntu_624498".split() + \
			"superuser_356658 superuser_121201 superuser_455589 superuser_38460 superuser_739955 superuser_931151".split() + \
			"superuser_291105 superuser_627439 superuser_584013 superuser_399182 superuser_632675 superuser_706347".split() + \
			"superuser_670748 superuser_369878 superuser_830279 superuser_927242 superuser_850786".split()

def get_annotations(line):
	set_info, post_id, best, valids, confidence = line.split(',')
	annotator_name = set_info.split('_')[0]
	sitename = set_info.split('_')[1]
	best = int(best)
	valids = [int(v) for v in valids.split()]
	confidence = int(confidence)
	return post_id, annotator_name, sitename, best, valids, confidence

def calculate_precision(model_pred_indices, best, valids):
	bp1, bp3, bp5 = 0., 0., 0.
	vp1, vp3, vp5 = 0., 0., 0.
	bp1 = len(set(model_pred_indices[:1]).intersection(set(best)))*1.0
	bp3 = len(set(model_pred_indices[:3]).intersection(set(best)))*1.0/3
	bp5 = len(set(model_pred_indices[:5]).intersection(set(best)))*1.0/5

	vp1 = len(set(model_pred_indices[:1]).intersection(set(valids)))*1.0
	vp3 = len(set(model_pred_indices[:3]).intersection(set(valids)))*1.0/3
	vp5 = len(set(model_pred_indices[:5]).intersection(set(valids)))*1.0/5
	return bp1, bp3, bp5, vp1, vp3, vp5

def calculate_avg_precision(model_probs, best, valids):
	best_bool = [0]*10
	valids_bool = [0]*10
	for i in range(10):
		if i in best:
			best_bool[i] = 1
		if i in valids:
			valids_bool[i] = 1
	bap = average_precision_score(best_bool, model_probs)
	if 1 in valids_bool:
		vap = average_precision_score(valids_bool, model_probs)
	else:
		vap = 0.
	if 1 in best_bool[1:]:
		bap_on9 = average_precision_score(best_bool[1:], model_probs[1:])
	else:
		bap_on9 = 0.
	if 1 in valids_bool[1:]:
		vap_on9 = average_precision_score(valids_bool[1:], model_probs[1:])
	else:
		vap_on9 = 0.
	return bap, vap, bap_on9, vap_on9	

def get_pred_indices(model_predictions, asc=False):
	preds = np.array(model_predictions)
	pred_indices = np.argsort(preds)
	if not asc:
		pred_indices = pred_indices[::-1] #since ascending sort and we want descending
	return pred_indices

def convert_to_probalitites(model_predictions):
	tot = sum(model_predictions)
	model_probs = [0]*10
	for i,v in enumerate(model_predictions):
		model_probs[i] = v*1./tot
	return model_probs

def evaluate_model_on_org(model_predictions, asc=False):
	br1_tot, br3_tot, br5_tot = 0., 0., 0.
	for post_id in model_predictions:
		model_probs = convert_to_probalitites(model_predictions[post_id])
		model_pred_indices = get_pred_indices(model_probs, asc)
		br1, br3, br5, vr1, vr3, vr5 = calculate_precision(model_pred_indices, [0], [0])
		br1_tot += br1	
		br3_tot += br3	
		br5_tot += br5
	N = len(model_predictions)
	return br1_tot*100.0/N, br3_tot*100.0/N, br5_tot*100.0/N	

def read_human_annotations(human_annotations_filename):
	human_annotations_file = open(human_annotations_filename, 'r')
	annotations = {}
	for line in human_annotations_file.readlines():
		line = line.strip('\n')
		splits = line.split('\t')
		post_id1, annotator_name1, sitename1, best1, valids1, confidence1 = get_annotations(splits[0])
		post_id2, annotator_name2, sitename2, best2, valids2, confidence2 = get_annotations(splits[1])		
		assert(sitename1 == sitename2)
		assert(post_id1 == post_id2)
		post_id = sitename1+'_'+post_id1
		best_union = list(set([best1, best2]))
		valids_inter = list(set(valids1).intersection(set(valids2)))
		annotations[post_id] = (best_union, valids_inter)
	return annotations

def evaluate_model(human_annotations_filename, model_predictions, asc=False):
	human_annotations_file = open(human_annotations_filename, 'r')
	br1_tot, br3_tot, br5_tot = 0., 0., 0.
	vr1_tot, vr3_tot, vr5_tot = 0., 0., 0.
	br1_on9_tot, br3_on9_tot, br5_on9_tot = 0., 0., 0.
	vr1_on9_tot, vr3_on9_tot, vr5_on9_tot = 0., 0., 0.
	bap_tot, vap_tot = 0., 0.
	bap_on9_tot, vap_on9_tot = 0., 0.
	N = 0
	for line in human_annotations_file.readlines():
		line = line.strip('\n')
		splits = line.split('\t')
		post_id1, annotator_name1, sitename1, best1, valids1, confidence1 = get_annotations(splits[0])
		post_id2, annotator_name2, sitename2, best2, valids2, confidence2 = get_annotations(splits[1])		
		assert(sitename1 == sitename2)
		assert(post_id1 == post_id2)
		post_id = sitename1+'_'+post_id1
		if post_id in BAD_QUESTIONS:
			continue
		best_union = list(set([best1, best2]))
		valids_inter = list(set(valids1).intersection(set(valids2)))
		valids_union = list(set(valids1+valids2))

		model_probs = convert_to_probalitites(model_predictions[post_id])	
		model_pred_indices = get_pred_indices(model_probs, asc)

		br1, br3, br5, vr1, vr3, vr5 = calculate_precision(model_pred_indices, best_union, valids_inter)
		br1_tot += br1	
		br3_tot += br3	
		br5_tot += br5	
		vr1_tot += vr1	
		vr3_tot += vr3	
		vr5_tot += vr5
		bap, vap, bap_on9, vap_on9 = calculate_avg_precision(model_probs, best_union, valids_inter)
		bap_tot += bap
		vap_tot += vap
		bap_on9_tot += bap_on9
		vap_on9_tot += vap_on9
	
		model_pred_indices = np.delete(model_pred_indices, 0)
		
		br1_on9, br3_on9, br5_on9, vr1_on9, vr3_on9, vr5_on9 = calculate_precision(model_pred_indices, best_union, valids_inter)
		
		br1_on9_tot += br1_on9	
		br3_on9_tot += br3_on9	
		br5_on9_tot += br5_on9	
		vr1_on9_tot += vr1_on9	
		vr3_on9_tot += vr3_on9
		vr5_on9_tot += vr5_on9

		N += 1
	
	human_annotations_file.close()
	return br1_tot*100.0/N, br3_tot*100.0/N, br5_tot*100.0/N, vr1_tot*100.0/N, vr3_tot*100.0/N, vr5_tot*100.0/N, \
			br1_on9_tot*100.0/N, br3_on9_tot*100.0/N, br5_on9_tot*100.0/N, vr1_on9_tot*100.0/N, vr3_on9_tot*100.0/N, vr5_on9_tot*100.0/N, \
			bap_tot*100./N, vap_tot*100./N, bap_on9_tot*100./N, vap_on9_tot*100./N

def read_model_predictions(model_predictions_file):
	model_predictions = {}
	for line in model_predictions_file.readlines():
		splits = line.strip('\n').split()
		post_id = splits[0][1:-2]
		predictions = [float(val) for val in splits[1:]]
		model_predictions[post_id] = predictions
	return model_predictions

def print_numbers(br1, br3, br5, vr1, vr3, vr5, br1_on9, br3_on9, br5_on9, vr1_on9, vr3_on9, vr5_on9, bmap, vmap, bmap_on9, vmap_on9, br1_org, br3_org, br5_org):
	print 'Best'
	print 'p@1 %.1f' % (br1)
	print 'p@3 %.1f' % (br3)
	print 'p@5 %.1f' % (br5)
	print 'map %.1f' % (bmap)
	print
	print 'Valid'
	print 'p@1 %.1f' % (vr1)
	print 'p@3 %.1f' % (vr3)
	print 'p@5 %.1f' % (vr5)
	print 'map %.1f' % (vmap)
	print
	print 'Best on 9'
	print 'p@1 %.1f' % (br1_on9)
	print 'p@3 %.1f' % (br3_on9)
	print 'p@5 %.1f' % (br5_on9)
	print 'map %.1f' % (bmap_on9)
	print
	print 'Valid on 9'
	print 'p@1 %.1f' % (vr1_on9)
	print 'p@3 %.1f' % (vr3_on9)
	print 'p@5 %.1f' % (vr5_on9)
	print 'map %.1f' % (vmap_on9)
	print 
	print 'Original'
	print 'p@1 %.1f' % (br1_org)
	#print 'p@3 %.1f' % (br3_org)
	#print 'p@5 %.1f' % (br5_org)

def main(args):
	model_predictions_file = open(args.model_predictions_filename, 'r')
	asc=False
	model_predictions = read_model_predictions(model_predictions_file)
	br1, br3, br5, vr1, vr3, vr5, \
	br1_on9, br3_on9, br5_on9, \
	vr1_on9, vr3_on9, vr5_on9, \
	bmap, vmap, bmap_on9, vmap_on9 = evaluate_model(args.human_annotations_filename, model_predictions, asc)
	br1_org, br3_org, br5_org = evaluate_model_on_org(model_predictions)

	print_numbers(br1, br3, br5, vr1, vr3, vr5, \
					br1_on9, br3_on9, br5_on9, \
					vr1_on9, vr3_on9, vr5_on9, \
					bmap, vmap, bmap_on9, vmap_on9, \
					br1_org, br3_org, br5_org)

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--human_annotations_filename", type = str)
	argparser.add_argument("--model_predictions_filename", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)

