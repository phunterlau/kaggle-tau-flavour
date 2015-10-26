#average each result files
weight = [0.055,0.945] 
print sum(weight)
file_list = [
            'layer2_test_V4_(experimental).csv', #this one from NN has bad KS score, but good AUC boost
            'ub_xgboost_selection_submission.csv'
            #'finer_tuned_NN.csv' 
            ]
contents = []

for fname in file_list:
    fi = open(fname,'r')
    header = fi.readline()
    contents.append([tuple(line.strip().split(',')) for line in fi])
    fi.close()

fw = open('ensembled_ub_xg_with_055_layer2.csv','w')
fw.write(header)
for content in zip(*contents):
    values = [float(j) for i,j in content]
    ave = sum(w*i for w,i in zip(weight,values))
    #print content[0][0],ave
    if ave>1:
        ave=1.
    fw.write('%s,%f\n'%(content[0][0],ave))
fw.close()
