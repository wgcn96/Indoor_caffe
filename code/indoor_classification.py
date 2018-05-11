#-*- coding:utf-8 -*-

from function_dataBase import *
from function import *


db, cursor = connectdb()
'''
print '--------------------feature 1 (stn) svm-----------------------'
FEATURE1_train_data, train_label = read_feature(db,cursor,traintable,'FEATURE1',train_num)
FEATURE1_test_data, test_label = read_feature(db,cursor,testtable,'FEATURE1',test_num)
result = mySVM(FEATURE1_train_data,FEATURE1_test_data,train_label,test_label)
print 'feature 1 result is: ',result
'''


print '--------------------feature 2 (baseline) svm-----------------------'
fsvm(db=db, cursor=cursor, featurename='FEATURE2')



print '-------------------------feature 3 4 5 svm-----------------------'
fsvm(db=db,cursor=cursor,featurename='FEATURE3')
fsvm(db=db,cursor=cursor,featurename='FEATURE4')
fsvm(db=db,cursor=cursor,featurename='FEATURE5')


'''
# 3 + 4 + 5 pca
print '--------------------feature 6 (3+4+5)-----------------------'
print '从数据库取 FEATURE 6 数据...'
FEATURE6_train_data, train_label = read_feature(db, cursor, traintable, "FEATURE6", train_num)
FEATURE6_test_data, test_label = read_feature(db, cursor, testtable, "FEATURE6", test_num)
print '训练SVM并测试...'
print 'the dimen is :'
print FEATURE6_train_data.shape
print FEATURE6_test_data.shape
FEATURE6_train_data_pca,FEATURE6_test_data_pca = myPCA(FEATURE6_train_data, FEATURE6_test_data,4096)
print FEATURE6_train_data_pca.shape
print FEATURE6_test_data_pca.shape
result = mySVM(FEATURE6_train_data_pca, FEATURE6_test_data_pca, train_label, test_label)
print 'pca feature 6 result is: ',result
'''
print '--------------------feature 6 (3+4+5)-----------------------'
print '从数据库取 FEATURE 6 数据...'
FEATURE3_train_data, train_label = read_feature(db, cursor, table_name=traintable, featurename="FEATURE3",
                                                num=train_num)
FEATURE3_test_data, test_label = read_feature(db, cursor, table_name=testtable, featurename="FEATURE3",
                                              num=test_num)
FEATURE4_train_data, train_label = read_feature(db, cursor, table_name=traintable, featurename="FEATURE4",
                                                num=train_num)
FEATURE4_test_data, test_label = read_feature(db, cursor, table_name=testtable, featurename="FEATURE4",
                                              num=test_num)
FEATURE5_train_data, train_label = read_feature(db, cursor, table_name=traintable, featurename="FEATURE5",
                                                num=train_num)
FEATURE5_test_data, test_label = read_feature(db, cursor, table_name=testtable, featurename="FEATURE5",
                                              num=test_num)

FEATURE6_train_data = np.concatenate((FEATURE3_train_data, FEATURE4_train_data, FEATURE5_train_data), 1)
FEATURE6_test_data = np.concatenate((FEATURE3_test_data, FEATURE4_test_data, FEATURE5_test_data), 1)
FEATURE6_train_data_pca,FEATURE6_test_data_pca = myPCA(FEATURE6_train_data, FEATURE6_test_data)
print FEATURE6_train_data_pca.shape
print FEATURE6_test_data_pca.shape
result = mySVM(FEATURE6_train_data_pca, FEATURE6_test_data_pca, train_label, test_label)
print 'pca feature 6 result is: ',result
'''
print '----------------------cam feature 8-----------------------'
FEATURE8_train_data, train_label = read_feature(db, cursor, traintable, "FEATURE8", train_num)
FEATURE8_test_data, test_label = read_feature(db, cursor, testtable, "FEATURE8", test_num)
result = mySVM(FEATURE8_train_data, FEATURE8_test_data, train_label, test_label)
print 'feature 8 result is: ',result


# 3 + 4 + 5 + cam
print '----------------------"1+2" + cam -------------------------'
FEATURE9_train_data = np.concatenate((FEATURE6_train_data, FEATURE8_train_data), 1)
FEATURE9_test_data = np.concatenate((FEATURE6_test_data, FEATURE8_test_data), 1)
print 'the dimen is :'
print FEATURE9_train_data.shape
print FEATURE9_test_data.shape
#result = mySVM(FEATURE9_train_data, FEATURE9_test_data, train_label, test_label)
#print 'feature 9 result is: ',result


# 特征融合 pca
FEATURE10_train_data,FEATURE10_test_data = myPCA(FEATURE9_train_data,FEATURE9_test_data)
print 'after pca , the dime is:'
print FEATURE10_train_data.shape
print FEATURE10_test_data.shape
result = mySVM(FEATURE10_train_data, FEATURE10_test_data, train_label, test_label)
print 'the result is: ',result


# lsq 特征融合 pca
print '---------------------lsq  "1+2" + cam ----------------------'
FEATURE7_train_data, train_label = read_feature(db, cursor, traintable, "FEATURE7", train_num)
FEATURE7_test_data, test_label = read_feature(db, cursor, testtable, "FEATURE7", test_num)
result = mySVM(FEATURE7_train_data, FEATURE7_test_data, train_label, test_label)
print 'lsq_cam result is :',result
FEATURE11_train_data = np.concatenate((FEATURE6_train_data, FEATURE7_train_data), 1)
FEATURE11_test_data = np.concatenate((FEATURE6_test_data, FEATURE7_test_data), 1)
print 'the dimen is :'
print FEATURE11_train_data.shape
print FEATURE11_test_data.shape
#result = mySVM(FEATURE11_train_data, FEATURE11_test_data, train_label, test_label)
FEATURE12_train_data,FEATURE12_test_data = myPCA(FEATURE11_train_data,FEATURE11_test_data)
result = mySVM(FEATURE12_train_data, FEATURE12_test_data, train_label, test_label)
print 'feature 12 result is: ',result

# stn + '1+2'
print '----------------------stn + "1+2" -------------------------'
FEATURE13_train_data = np.concatenate((FEATURE1_train_data, FEATURE6_train_data), 1)
FEATURE13_test_data = np.concatenate((FEATURE1_test_data, FEATURE6_test_data), 1)
FEATURE13_train_data,FEATURE13_test_data = myPCA(FEATURE13_train_data,FEATURE13_test_data)
result = mySVM(FEATURE13_train_data, FEATURE13_test_data, train_label, test_label)
print 'feature 13 result is: ',result
'''
closedb(db)
