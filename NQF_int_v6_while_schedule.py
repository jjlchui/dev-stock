import requests 
import datetime
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
import re 
import os
import time
import os.path
import feather
import datetime
import pytz
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from apscheduler.schedulers.blocking import BlockingScheduler

def usny_curtime():
    nyc_datetime = datetime.now(pytz.timezone('America/New_York'))
    fmt = '%Y-%m-%d %H:%M:%S'
    time_stamp = nyc_datetime.strftime(fmt)
    return time_stamp
"""
###### get NDX start ######
def get_price_stock(ticker="^NDX"):
    x=0
    try:
        url = "https://finance.yahoo.com/quote/"+ticker+"?p="+ticker+"&.tsrc=fin-srch"
        #url = "https://finance.yahoo.com/quote/U?p=U&.tsrc=fin-srch"
        headers = {"User-Agent" : "Chrome/101.0.4951.41"}
        r = requests.get(url, headers=headers)
        page_content = r.content
        #soup = BeautifulSoup(page_content, 'lxml')
        soup = BeautifulSoup(page_content, "html.parser")
        web_content = soup.find('div', {'class' :'D(ib) Mend(20px)'})

        if (web_content == None):
            time.sleep(1)
            return 0,0,0,1
        else:
            stock_price = web_content.find("fin-streamer", {'class' : 'Fw(b) Fz(36px) Mb(-4px) D(ib)'}).text
            if(stock_price == ""  ):
                time.sleep(1)
                return 0,0,0,1
            else:
                stock_price = stock_price.replace(",", "")
                print('stock price :', stock_price)
                change = web_content.find("fin-streamer", {'data-field' : "regularMarketChangePercent"}).text
                tabl = soup.find_all("div", {'class' : "D(ib) W(1/2) Bxz(bb) Pend(12px) Va(t) ie-7_D(i) smartphone_D(b) smartphone_W(100%) smartphone_Pend(0px) smartphone_BdY smartphone_Bdc($seperatorColor)"})
                change = change.strip('()')
                change = re.sub('%', "", change)
                print('% change :',change)
                
                web_content1 = soup.find('table', {'class' :"W(100%) M(0) Bdcl(c)"}).text
                words = web_content1.split()
                old, new = words[-1].split('e')
                new_vol = new.split('A')
                my_var = new_vol[0]
                print('new_vol', my_var)
                if (my_var == "N/"):
                    volume = 0
                else:
                    volume = my_var
                    volume = volume.replace(",", "")
                print("volume: ",volume)
                Error = 0
                return stock_price, change, volume, Error
    except ConnectionError:
        print("Network Issue !!!")
        stock_price = 0
        change = 0
        volume = 0
        Error = 1
        print("Network Issue !!!")
        
    return stock_price, change, volume, Error
###### get NDX end ######

"""
def get_price_stock(ticker="NQ=F"):
    
    disable_warnings(InsecureRequestWarning)
    try:
        start_time = time.time()
        print("*******START TIME(2)", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        url = "https://finance.yahoo.com/quote/"+ticker+"?p="+ticker+"&.tsrc=fin-srch"
        #r=requests.get("https://finance.yahoo.com/quote/"+ticker+"?p="+ticker+"&.tsrc=fin-srch")
        #url = "https://finance.yahoo.com/quote/U?p=U&.tsrc=fin-srch"
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
        r = requests.get(url, headers=headers, stream=True, verify=False)
        ##### new add to handle chunk_size 
          
        page_content = r.content
        #soup = BeautifulSoup(page_content, 'lxml')
        print('beautiful soup ------')
        soup = BeautifulSoup(r.content, "html.parser")
        print('soup----')
        web_content = soup.find('div', {'class' :'D(ib) Mend(20px)'}) 
        print('web_content----')


        if (web_content == None):
            time.sleep(1)
            return 0,0,0,0
        else:
            print('web_content :')
            stock_price1 = web_content.find("fin-streamer", {'class' : 'Fw(b) Fz(36px) Mb(-4px) D(ib)'}).text
            print('stock price1 :', stock_price1) 
            if(stock_price1 == ""  ):
                time.sleep(1)
                return 0,0,0,0
            else:                
                stock_price = stock_price1.replace(",", "")
                print('stock price :', stock_price)
                change = web_content.find("fin-streamer", {'data-field' : "regularMarketChangePercent"}).text
                #tabl = soup.find_all("div", {'class' : "D(ib) W(1/2) Bxz(bb) Pend(12px) Va(t) ie-7_D(i) smartphone_D(b) smartphone_W(100%) smartphone_Pend(0px) smartphone_BdY smartphone_Bdc($seperatorColor)"})
                change = change.strip('()')
                change = re.sub('%', "", change)
                print('% change :',change)
                
                web_content1 = soup.find('table', {'class' :"W(100%) M(0) Bdcl(c)"}).text
                print('web_content1....', web_content1) 
                words = web_content1.split()
                print('words.....', words)
                old, new = words[4].split('e')
                new_vol = new.split('A')
                my_var = new_vol[0]
                print('new_vol', my_var)
                if (my_var == "N/"):
                    volume = 0
                else:
                    volume = my_var
                    volume = volume.replace(",", "")
                print("volume: ",volume)
                print("*******START TIME(3)", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
                Error = 0
                return stock_price, change, volume , Error   
    except requests.exceptions.ConnectionError:
        r="No response"
        stock_price = 0
        change = 0
        volume = 0
        Error = 1
        print('Connect Error')
    except requests.exceptions.Timeout:
        stock_price = 0
        change = 0
        volume = 0
        Error = 1
        #status code 408
        print("Timeout Error ocurred, program waits and retries again")
        time.sleep(1)
    except requests.exceptions.TooManyRedirects:
        stock_price = 0
        change = 0
        volume = 0
        Error = 1
        #status code 301
        print("Too many redirects")
        raise SystemExit()
    except requests.exceptions.HTTPError as e:
        stock_price = 0
        change = 0
        volume = 0
        Error = 1
        #overally status codes 400,500, ...
        print("HTTP error, status code is "+ str(r.status_code)+
            "\nMessage from Server: "+r.content.decode("utf-8") )
        raise SystemExit()
    except requests.exceptions.RequestException as e:
        stock_price = 0
        change = 0
        volume = 0
        Error = 1
        print(e)
        raise SystemExit()
        print("Network Issue !!!")
        
    return stock_price, change, volume , Error     


#a = get_price_stock('NQ=F')

def Save_csv(file, filename, mode):  
    cwd = os.getcwd()
    path = os.path.dirname(cwd)

    file_path = path + "\\stock\\data\\"
    #time_stamp = usny_curtime()

    time_stamp = datetime.datetime.now() .strftime('%Y-%m-%d %H:%M:%S')
    timefile = os.path.join(file_path + str(time_stamp[0:11]) +"NQ=F USTime" + filename)
    
    if mode=='w':
        file.to_csv(timefile, mode='w', header=False, index=False, encoding = 'utf-8')
    else:
        file.to_csv(timefile, mode='a', header=False, index=False, encoding = 'utf-8')
    print("..........save_csv", timefile)
    return timefile, time_stamp

def Save_feather(file, filename):  
    cwd = os.getcwd()
    path = os.path.dirname(cwd)

    file_path = path + "\\stock\\data\\"
    #time_stamp = usny_curtime()

    time_stamp = datetime.datetime.now() .strftime('%Y-%m-%d %H:%M:%S')
    timefile = os.path.join(file_path + str(time_stamp[0:11]) +"NQ=F USTime" + filename)

    file.to_feather(timefile)
    print("..........save_feather", timefile)
    return timefile, time_stamp

#sched = BlockingScheduler(timezone='America/New_York')
#@sched.scheduled_job("cron", hour="8-17", minute = "*", )
#sched = BlockingScheduler()
#@sched.add_job('cron', hour='0-23', minute = '*')

def get_stock_data():

    Running = True  
    i = 0  
    while (Running):
        start_time = time.time()
        print("*******START TIME(1)", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        
        #time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        price, change, volume, Error = get_price_stock("NQ=F")
        
        if Error == 1:
            price, change, volume, Error = get_price_stock("NQ=F")
        
        
        if (i < 800):
            if (float(price) == 0 ):
                print('price, == empyt')
                pass  
            else:
                info = []
                info.append(price)
                info.append(change)
                info.append(volume)
                time_stamp = datetime.datetime.now() .strftime('%Y-%m-%d %H:%M:%S')
                #time_stamp = usny_curtime()
                col = []
                col = [time_stamp]
                col.extend(info) 
                print("-----col1", col)
            #time.sleep(5)
            #time.sleep(60.0 - ((time.time() - start_time) % 60.0))
            print('time', time.time(), start_time)
        else:
            Running = False
            
    
        if (float(price) == 0):
                pass    
        else:
            df = pd.DataFrame(col)
            df = df.T
            print("-----col2", col)
    
        # OUTPUT : _reconcil_stock_data.csv
        
        outputfile, time_stamp = Save_csv(df, '_reconcil_stock_data.csv', 'a')   
    
        #data = Process_Data(df)
        
        # OUTPUT : _out_stock_data.csv
        data=pd.read_csv(outputfile)
        data.columns = ['datetime', 'price', 'change', 'volume']
        if (price != "NaN"):
            data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
            data['price'] = pd.to_numeric(data['price'])
            data['volume'] = pd.to_numeric(data['volume'])
            data.set_index("datetime", inplace=True)
            data_vol=data.copy()
            print("***********before candle", data)
            data= data['price'].resample('1Min').agg({'open': 'first', 
                                         'high': 'max', 
                                         'low': 'min', 
                                         'close': 'last'})
            print("***********after candle", data, )
            #d_vol.set_index(['datetime'], inplace=True)
            d_vol = data_vol['volume'].resample('1Min').mean()
            data['volumne']=d_vol
            #data = data[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            data.fillna(method="bfill" , inplace=True)
            data.reset_index(inplace=True)
            
            
            outputfile, time_stamp = Save_csv(data, '_out_stock_data.csv', 'w' )
            outputfile_feather, time_stamp = Save_feather(data, '_feather_stock_data.feather')
        
        end_time = time.time()
        print("*******END TIME(2)", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))

    
#sched = BlockingScheduler(timezone='America/New_York')
#job start 9:am to 5:pm
sched = BlockingScheduler()
sched.add_job(get_stock_data, 'cron', hour='19-20', minute = '*')
sched.start() 




       
