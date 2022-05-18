#File to transform the spam-sms row-streaming dataset to evolving 

with open('Data/streaming-data/spam-sms') as f:
    lines = f.readlines()

import re

graph = {}
positive_ids = []
negative_ids = []

X_file = open("Data/Evolving-data/spam-sms-evolving", "w")
y_file = open("Data/Evolving-data/labels-spam-sms-evolving", "w")

for i, f_list in enumerate(lines):

  if i not in graph.keys():
    graph[i] = []
  
  split_line = f_list.split() 

  y = int(split_line[0])

  if y == 0:
    negative_ids.append(i)
  else:
    positive_ids.append(i)

  y_file.write(str(y) + '\n')

  prev = None

  for j, f in enumerate(split_line[1:]):
    
    #Add current
    X_file.write(str(i) + ' '+ f + '\n')

    #Check if less than
    f_name = f.split(":")

    id_feature = int(f_name[0]) - 1
    
    if j == 0:
      while id_feature >= 0:
        X_file.write(str(i) + ' '+ str(id_feature) + ':0' +  '\n')
        id_feature -=1
    else:
      while id_feature > prev:
        X_file.write(str(i) + ' '+ str(id_feature) + ':0' +  '\n')
        id_feature -=1
    
    prev = int(f_name[0]) 

X_file.close()
y_file.close()
