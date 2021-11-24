from flask import Flask
app = Flask(__name__)

##A database can be initialized here if we decide we want one
##Im using mongoDB in another class and its working well with this 
# I can do an implementation in the future

from app import routes