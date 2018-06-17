import sys
from helper import *
from collections import defaultdict
from difflib import SequenceMatcher
import pdb

class PostQuesAns:

	def __init__(self, post_title, post, post_sents, question_comment, answer):
		self.post_title = post_title
		self.post = post
		self.post_sents = post_sents
		self.question_comment = question_comment
		self.answer = answer

class PostQuesAnsGenerator:

	def __init__(self):
		self.post_ques_ans_dict = defaultdict(PostQuesAns)

	def get_diff(self, initial, final):
		s = SequenceMatcher(None, initial, final)
		diff = None
		for tag, i1, i2, j1, j2 in s.get_opcodes():
			if tag == 'insert':
				diff = final[j1:j2]
		if not diff:
			return None
		return diff		

	def find_first_question(self, answer, question_comment_candidates, vocab, word_embeddings):
		first_question = None
		first_date = None
		for question_comment in question_comment_candidates:
			if first_question == None:
				first_question = question_comment
				first_date = question_comment.creation_date
			else:
				if question_comment.creation_date < first_date:
					first_question = question_comment
					first_date = question_comment.creation_date
		return first_question

	def find_answer_comment(self, all_comments, question_comment, post_userId):
		answer_comment, answer_comment_date = None, None
		for comment in all_comments:
			if comment.userId and comment.userId == post_userId:
				if comment.creation_date > question_comment.creation_date:
					if not answer_comment or (comment.creation_date < answer_comment_date):
						answer_comment = comment
						answer_comment_date = comment.creation_date
		return answer_comment

	def generate_using_comments(self, posts, question_comments, all_comments, vocab, word_embeddings):
		for postId in posts.keys():
			if postId in self.post_ques_ans_dict.keys():
				continue
			if posts[postId].typeId != 1: # is not a main post
				continue
			first_question = None
			first_date = None
			for question_comment in question_comments[postId]:
				if question_comment.userId and question_comment.userId == posts[postId].owner_userId:
					continue #Ignore comments by the original author of the post
				if first_question == None:
					first_question = question_comment
					first_date = question_comment.creation_date
				else:
					if question_comment.creation_date < first_date:
						first_question = question_comment
						first_date = question_comment.creation_date
			question = first_question
			if not question:
				continue
			answer_comment = self.find_answer_comment(all_comments[postId], question, posts[postId].owner_userId)	
			if not answer_comment:
				continue
			answer = answer_comment
			self.post_ques_ans_dict[postId] = PostQuesAns(posts[postId].title, posts[postId].body, \
															posts[postId].sents, question.text, answer.text)

	def generate(self, posts, question_comments, all_comments, posthistories, vocab, word_embeddings):
		for postId, posthistory in posthistories.iteritems():
			if not posthistory.edited_posts:
				continue
			if posts[postId].typeId != 1: # is not a main post
				continue
			if not posthistory.initial_post:
				continue
			first_edit_date, first_question, first_answer = None, None, None
			for i in range(len(posthistory.edited_posts)):
				answer = self.get_diff(posthistory.initial_post, posthistory.edited_posts[i])
				if not answer:
					continue
				else:
					answer = remove_urls(' '.join(answer))
					answer = answer.split()
					if is_too_short_or_long(answer):
						continue
				question_comment_candidates = []
				for comment in question_comments[postId]:
					if comment.userId and comment.userId == posts[postId].owner_userId:
						continue #Ignore comments by the original author of the post
					if comment.creation_date > posthistory.edit_dates[i]:
						continue #Ignore comments added after the edit
					else:
						question_comment_candidates.append(comment)
				question = self.find_first_question(answer, question_comment_candidates, vocab, word_embeddings)
				if not question:
					continue
				answer_comment = self.find_answer_comment(all_comments[postId], question, posts[postId].owner_userId)
				if answer_comment and 'edit' not in answer_comment.text: #prefer edit if comment points to the edit
					question_indices = get_indices(question.text, vocab)
					answer_indices = get_indices(answer, vocab)
					answer_comment_indices = get_indices(answer_comment.text, vocab)
					if get_similarity(question_indices, answer_comment_indices, word_embeddings) > get_similarity(question_indices, answer_indices, word_embeddings):
						answer = answer_comment.text
				if first_edit_date == None or posthistory.edit_dates[i] < first_edit_date:
					first_question, first_answer, first_edit_date = question, answer, posthistory.edit_dates[i]

			if not first_question:
				continue 
			self.post_ques_ans_dict[postId] = PostQuesAns(posts[postId].title, posthistory.initial_post, \
															posthistory.initial_post_sents, first_question.text, first_answer)

		self.generate_using_comments(posts, question_comments, all_comments, vocab, word_embeddings)
		return self.post_ques_ans_dict
