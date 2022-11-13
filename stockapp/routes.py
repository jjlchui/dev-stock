from stockapp import app
from flask import Flask, render_template, session, redirect
from flask import request
from plotlydash import test
from plotlydash import Simple_gui
from plotlydash import Copymacd
from plotlydash import Max_Min_line
from plotlydash import macd_ma
from plotlydash import RegressionRoll
from plotlydash import Renko
from plotlydash import BBand
import os  
import time


BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'stockapp/static/upload/')
test.getLayout(app)
Simple_gui.create_dash(app)
Copymacd.create_dash(app)
Max_Min_line.create_dash(app)
macd_ma.create_dash(app)
RegressionRoll.create_dash(app)
Renko.create_dash(app)
BBand.create_dash(app)



#NQF_group_test6_csv.get_price_stock(ticker="NQ=F")

@app.route("/", methods=['GET','POST'])
def index():
    
    if  request.method == "POST":
        upload_file = request.files['file_name']
        filename = upload_file.filename
        print('The filename is', filename)
        ext = filename.split('.')[-1]
        print('The extension of file', ext)
        print('path', UPLOAD_PATH)
        if ext.lower() in ['csv']:
            print('path', UPLOAD_PATH)
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(path_save)
            print('File upload sucessfully')
    else:
        print('Use only csv file')
        
    return render_template("upload.html", title='Home')



    


@app.errorhandler(404)
def handling_page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404





