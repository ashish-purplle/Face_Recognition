from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import boto3
import os
s3 = boto3.resource('s3')


try:
    s3.create_bucket(Bucket=os.environ.get("S3_BUCKET_NAME"), CreateBucketConfiguration={
            'LocationConstraint': 'us-west-2'
        },)
except Exception as e:
    print "bucket already exits"



def uploadToS3(data,filename):
    try:
        s3.upload_file(data, os.environ.get("S3_BUCKET_NAME"), filename)
        print "file uploaded to s3 successfully"
    except Exception as e:
        print "error in uploading file"