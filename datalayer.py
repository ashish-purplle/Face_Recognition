from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra import ConsistencyLevel
from logger import logger
import os



auth_provider = PlainTextAuthProvider(username=os.environ.get("CASSANDRA_USER"), password=os.environ.get("CASSANDRA_PASSWORD"))
cluster = Cluster(os.environ.get("CASSANDRA_SERVER").split(","),port=9042,auth_provider=auth_provider)
session = cluster.connect(os.environ.get("CASSANDRA_KEYSPACE"))

getQuery = session.prepare("SELECT * from name_by_class where class_id =?")
getQuery.consistency_level = ConsistencyLevel.QUORUM


getQuery.consistency_level = ConsistencyLevel.QUORUM

def getData(classId):
    try:
        logger.info('inside getData function try block')
        rows = session.execute(query=getQuery, parameters=[classId])
        logger.info('inside getData function try block data fetched successfully')
        return rows
    except Exception as e:
        logger.error('error while getting the data',exc_info=True)










