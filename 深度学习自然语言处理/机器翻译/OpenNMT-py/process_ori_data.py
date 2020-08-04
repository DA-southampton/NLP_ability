file=open('train_data.csv','r')
lines=file.readlines()

src_train=open('src-train.txt','w')
tgt_train=open('tgt-train.txt','w')

src_val=open('src-val.txt','w')
tgt_val=open('tgt-val.txt','w')

chinese_lists=[]
english_lists=[]
index=0
for line in lines:
    if index ==0:
        index+=1
        continue
    line=line.strip().split(',')
    chinese=line[1].strip().split('_')
    english=line[2].strip().split('_')
    chinese_lists.append(' '.join(chinese))
    english_lists.append(' '.join(english))
    index+=1
    

assert len(chinese_lists)==len(english_lists)

split_num=int(0.85*index)

for num in range(len(english_lists)):
    if num<=split_num:
        src_train.write(chinese_lists[num]+'\n')
        tgt_train.write(english_lists[num]+'\n')
    else:
        src_val.write(chinese_lists[num]+'\n')
        tgt_val.write(english_lists[num]+'\n')

src_train.close()
tgt_train.close()

src_val.close()
tgt_val.close()







file=open('test_cs_a.csv','r')
lines=file.readlines()

src_test=open('src-test.txt','w')

for line in lines:
    line=line.strip().split(',')
    cont=line[2].split('_')
    cont=' '.join(cont)
    src_test.write(cont+'\n')
src_test.close()