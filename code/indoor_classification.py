#-*- coding:utf-8 -*-

from function_dataBase import *
from function import *


db, cursor = connectdb()



print '--------------------feature 2 (baseline) svm-----------------------'
fsvm(db=db, cursor=cursor, featurename='FEATURE2')



print '-------------------------feature 3 4 5 svm-----------------------'
fsvm(db=db,cursor=cursor,featurename='FEATURE3')
fsvm(db=db,cursor=cursor,featurename='FEATURE4')
fsvm(db=db,cursor=cursor,featurename='FEATURE5')



# 3 + 4 + 5 pca
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

print '----------------------cam feature 8-----------------------'
FEATURE8_train_data, train_label = read_feature(db, cursor, traintable, "FEATURE8", train_num)
FEATURE8_test_data, test_label = read_feature(db, cursor, testtable, "FEATURE8", test_num)
result = mySVM(FEATURE8_train_data, FEATURE8_test_data, train_label, test_label)
print 'feature 8 result is: ',result


closedb(db)
