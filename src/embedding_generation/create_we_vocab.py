import sys
import cPickle as p

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "usage: python create_we_vocab.py <word_vectors.txt> <output_we.p> <output_vocab.p>"
		sys.exit(0)
	word_vectors_file = open(sys.argv[1], 'r')
	word_embeddings = []
	vocab = {}
	i = 0
	for line in word_vectors_file.readlines():
		vals = line.rstrip().split(' ')
		vocab[vals[0]] = i
		word_embeddings.append(map(float, vals[1:]))
		i += 1
	p.dump(word_embeddings, open(sys.argv[2], 'wb'))
	p.dump(vocab, open(sys.argv[3], 'wb'))

