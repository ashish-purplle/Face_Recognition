from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra import ConsistencyLevel
from logger import logger
import os
import time

auth_provider = PlainTextAuthProvider(username=os.environ.get("CASSANDRA_USER"),
                                      password=os.environ.get("CASSANDRA_PASSWORD"))
cluster = Cluster(os.environ.get("CASSANDRA_SERVER").split(","), port=9042, auth_provider=auth_provider)
session = cluster.connect(os.environ.get("CASSANDRA_KEYSPACE"))

getQuery = session.prepare("SELECT * from images_by_class where class_id =?")
getQuery.consistency_level = ConsistencyLevel.QUORUM

insertClassInfoQuery = session.prepare("INSERT INTO classinfo(class_id,created_date)VALUES (?,?)")

insertImageByClassQuery = session.prepare("INSERT INTO images_by_class(class_id,camera_id,location,s3url,created_date)VALUES (?,?,?,?,?)")


def getData(classId):
    try:
        logger.info('inside getData function try block')
        rows = session.execute(query=getQuery, parameters=[classId])
        logger.info('inside getData function try block data fetched successfully')
        return rows
    except Exception as e:
        logger.error('error while getting the data', exc_info=True)


def insertDataInClassInfo(data):


    try:
        current_time = time.time()*1000
        logger.info('inside insertData function try block')
        rows = session.execute(query=insertClassInfoQuery, parameters=[data['classId'],current_time])
        logger.info('inside insertData function try block data fetched successfully')
        return rows
    except Exception as e:
        logger.error('error while inserting the data', exc_info=True)

def insertDataInImageByClass(data):


    try:
        current_time = time.time()*1000
        logger.info('inside insertData function try block')
        rows = session.execute(query=insertImageByClassQuery, parameters=[data['classId'],data['cameraId'],data['location'],data['s3url'],current_time])
        logger.info('inside insertData function try block data fetched successfully')
        return rows
    except Exception as e:
        logger.error('error while inserting the data', exc_info=True)
