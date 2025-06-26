import os
import sys
import pymongo
import certifi

from src.logger import logging
from src.exception import MyException
from src.constants import DATABASE_NAME , MONGODB_URL_KEY

ca= certifi.where()

class MongoDBClient:

    client = None

    def __init__(self , database_name : str = DATABASE_NAME) -> None:
        
        try:

            if MongoDBClient.client is None:
                mongo_db_url = MONGODB_URL_KEY
                if mongo_db_url is None:
                    raise Exception(f"Environment variable {MONGODB_URL_KEY} is missing.")
                
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url , tlsCAFile = ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("Mongodb connected successfully")

        except Exception as e:
            raise MyException(e , sys)
        

                 
