import boto3
import os
from src.constants import AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_ACCESS_KEY_ID_ENV_KEY, REGION_NAME
from dotenv import load_dotenv
load_dotenv()



class S3Client:

    s3_client=None
    s3_resource = None
    def __init__(self, region_name=REGION_NAME):
        """ 
        Create a connection with our aws environment 
        """

        if S3Client.s3_resource==None or S3Client.s3_client==None:
            __access_key_id =  AWS_ACCESS_KEY_ID_ENV_KEY
            __secret_access_key = AWS_SECRET_ACCESS_KEY_ENV_KEY
            if __access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not not set.")
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")
        
            S3Client.s3_resource = boto3.resource('s3',
                                            aws_access_key_id=__access_key_id,
                                            aws_secret_access_key=__secret_access_key,
                                            region_name=region_name
                                            )
            S3Client.s3_client = boto3.client('s3',
                                        aws_access_key_id=__access_key_id,
                                        aws_secret_access_key=__secret_access_key,
                                        region_name=region_name
                                        )
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client