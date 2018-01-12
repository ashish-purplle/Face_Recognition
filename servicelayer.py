from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
import boto3
s3 = boto3.resource('s3')

try:
    s3.create_bucket(Bucket=os.environ.get("S3_BUCKET_NAME"), CreateBucketConfiguration={
            'LocationConstraint': 'ap-south-1'
        },)
except Exception as e:
    print("bucket already exits")



def uploadToS3(data,filename):
    try:
        resp = s3.Bucket(os.environ.get("S3_BUCKET_NAME")).put_object(ACL='public-read',Key=filename, Body=data, ContentType='image/jpeg')
        main_img_url = os.environ.get("S3_HOST")+'/'+resp.bucket_name+'/'+resp.key
        return  main_img_url
        print("file uploaded to s3 successfully")
    except Exception as e:
        print("error in uploading file",e)