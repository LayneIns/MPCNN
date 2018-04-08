# coding: utf-8

import re
import sys


def readRelations(filepath):
	relation_list = []
	with open(filepath) as fin:
		for line in fin:
			line = line.decode("utf-8").strip()
			relation_list.append(line)
	return relation_list


def processData(filepath, relation_list, flag=False):
	data = []
	with open(filepath) as fin:
		for line in fin:
			line = line.decode("utf-8").strip()
			line_list = line.split("\t")
			golden = int(line_list[0])
			if line_list[1] != "noNegativeAnswer":
				negative = [int(num) for num in line_list[1].split() if num.strip()]
			else:
				negative = []
			if flag:
				negative.remove(golden)
			question = line_list[2]
			data.append([relation_list[golden-1], question, 1])
			for neg in negative:
				data.append([relation_list[neg-1], question, 0])
	return data


def writefile(data_list, filepath):
	with open(filepath, 'w') as fout:
		for item in data_list:
			line = item[1] + " $$ "
			line += item[0]
			for word in item[0].split("/"):
				if word.strip():
					line += (" " + word)
			line += " $$ "
			line += str(item[2])
			line += "\n"
			fout.write(line.encode("utf-8"))


if __name__ == "__main__":
	relation_list = readRelations("./relation.2M.list")
	training_data_list = processData("./train.replace_ne.withpool", relation_list)
	valid_data_list = processData("./valid.replace_ne.withpool", relation_list, True)
	test_data_list = processData("./test.replace_ne.withpool", relation_list, True)

	writefile(training_data_list, "../training.txt")
	writefile(test_data_list, "../testing.txt")
	writefile(valid_data_list, "../valid.txt")
