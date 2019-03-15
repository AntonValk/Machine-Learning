# COMP 551 Assignment 1
import numpy as np
import matplotlib.pyplot as plt

# Question 1
def generate_student_days(num_days):
	random_variable = np.random.uniform(0,1,num_days)
	event_log = np.zeros(4,dtype=int)
	for element in random_variable:
		if element <= 0.2:
			event_log[0] += 1
			# Case movies
		elif element <= 0.6:
			event_log[1] += 1
			# Case COMP-551
		elif element <= 0.7:
			event_log[2] += 1
			# Case playing
		else:
			event_log[3] += 1
			# Case studying
	print("Breakdown of student activities for", num_days ,"days.")
	print("Movies for", event_log[0], "days.")
	print("COMP-551 for", event_log[1], "days.")
	print("Playing for", event_log[2], "days.")
	print("Studying for", event_log[3], "days.")
	return event_log

def plot_bar(event_log):
	# this plots the bar graph
	label = ['Movies', 'COMP-551', 'Playing', 'Studying']
	index = np.arange(len(label))
	plt.bar(index, event_log)
	plt.xlabel('Activity', fontsize = 8)
	plt.ylabel('No of Days', fontsize = 8)
	plt.xticks(index, label, fontsize = 8, rotation = 30)
	plt.title('Daily Activity Histogram for {} days.'.format(sum(event_log)))
	plt.show()

if __name__ == "__main__":
	day_count = 1000
	log = generate_student_days(day_count)
	plot_bar(log)