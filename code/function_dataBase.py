# -*- coding:utf-8 -*-

import MySQLdb
import struct
import cPickle as pickle
import numpy as np

from function_data import *

traintable = "indoor67train"
testtable = "indoor67test"


def connectdb():
    db = MySQLdb.connect("127.0.0.1", "root", "123123", "indoor")
    cursor = db.cursor()
    return db, cursor


def closedb(db):
    db.close()


# =================创建数据表===============
# =========================================

def creattb(db, cursor, tbname, fileurl):
    sql = "CREATE TABLE " + tbname + " (\
            URL  CHAR(200) NOT NULL,\
            LABEL INT NOT NULL,  \
            ID INT NOT NULL AUTO_INCREMENT,\
            FEATURE1 BLOB, FEATURE2 BLOB, FEATURE3 BLOB, \
            FEATURE4 BLOB, FEATURE5 BLOB, FEATURE6 BLOB,\
            FEATURE7 BLOB, FEATURE8 BLOB, FEATURE9 BLOB,FEATURE10 BLOB,\
            primary key (ID))ENGINE=MyISAM"
    cursor.execute(sql)
    f = open(fileurl, "r")
    while True:
        line = f.readline()
        if line:
            line = line.strip()
            pos = line.find('/')
            label_str = line[:pos]
            lab = getLabel(label_str)
            sql = "INSERT INTO " + tbname + " (URL, LABEL) VALUES ('%s', '%d')" % (line, lab)
            try:
                cursor.execute(sql)
                db.commit()
            except:
                db.rollback()
        else:
            break
    f.close()


# ================================数据库接口============================
# ======================================================================
def write_feature_to_db(db, cursor, table_name, featurename, feature):
    blobstr = []
    num, feature_dim = feature.shape
    for i in range(num):
        result = ''
        for j in range(feature_dim):
            tmp = struct.pack('f', feature[i][j])
            result += tmp
        blobstr.append(result)

    sql = "UPDATE " + table_name + " SET " + featurename + " = %s WHERE ID  = %s"
    for i in range(num):
        rowID = i + 1
        args = (blobstr[i], str(rowID))
        try:
            cursor.execute(sql, args)
            db.commit()
        except Exception:
            print "write DB error"
            db.rollback()


def read_feature(db, cursor, table_name, featurename, num, labelFlag=True):
    blobstr = []
    feature = []
    label = []
    ori_sql = "SELECT " + featurename + ' , LABEL' + " FROM " + table_name + " WHERE ID = {}"
    for i in range(num):
        rowID = i + 1
        sql = ori_sql.format(rowID)
        cursor.execute(sql)
        tmp = cursor.fetchall()
        blobstr.append(tmp[0][0])
        label.append(tmp[0][1])
    for i in range(num):
        for j in range(len(blobstr[i]) / 4):
            result = struct.unpack('f', blobstr[i][j * 4:(j + 1) * 4])
            feature.append(result)
    feature = np.asarray(feature).astype('float32')
    feature = feature.reshape(num, feature.shape[0] / num)
    label = np.array(label).astype('int32')
    if labelFlag == True:
        return feature, label
    else:
        return feature


# ===================work script===================
if __name__ == '__main__':
    db, cursor = connectdb()
    # creattb(db,cursor,traintable,train_file_url)
    # creattb(db,cursor,testtable,test_file_url)
    closedb(db)
