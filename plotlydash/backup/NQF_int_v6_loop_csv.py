import requests 
import datetime
import pandas as pd
from bs4 import BeautifulSoup
import re 
import os
import time
import os.path
import feather
import pytz

def usny_curtime():
    nyc_datetime = datetime.now(pytz.timezone('America/New_York'))
    fmt = '%Y-%m-%d %H:%M:%S'
    time_stamp = nyc_datetime.strftime(fmt)
    return time_stamp

def get_price_stock(ticker="NQ=F"):


    try:
        start_time = time.time()
        print("*******START TIME(2)", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        url = "https://finance.yahoo.com/quote/"+ticker+"?p="+ticker+"&.tsrc=fin-srch"
        #url = "https://finance.yahoo.com/quote/U?p=U&.tsrc=fin-srch"
        headers = {"User-Agent" : "Chrome/101.0.4951.41"}
        r = requests.get(url, headers=headers)
        page_content = r.content
        #soup = BeautifulSoup(page_content, 'lxml')
        print('beautiful soup ------')
        soup = BeautifulSoup(page_content, "html.parser")
        web_content = soup.find('div', {'class' :'D(ib) Mend(20px)'}) 
        


        if (web_content == None):
            time.sleep(1)
            return 0,0,0
        else:
            print('web_content :')
            stock_price1 = web_content.find("fin-streamer", {'class' : 'Fw(b) Fz(36px) Mb(-4px) D(ib)'}).text
            print('stock price1 :', stock_price1) 
            if(stock_price1 == ""  ):
                time.sleep(1)
                return 0,0,0
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
    except ConnectionError:
        stock_price = 0
        change = 0
        volume = 0
        print("Network Issue !!!")
        
    return stock_price, change, volume      
    

#a = get_price_stock('NQ=F')

def Save_csv(file, filename, mode):  
    cwd = os.getcwd()
    path = os.path.dirname(cwd)

    file_path = path + "\\stock\\data\\"

    #time_stamp = datetime.datetime.now() - datetime.timedelta(hours=12)

    time_stamp = datetime.datetime.now() .strftime('%Y-%m-%d %H:%M:%S')
    timefile = os.path.join(file_path + str(time_stamp[0:11]) +" NQ=F USTime" + filename)
    #file.drop_duplicates(subset='time', keep=False)
    if mode=='w':
        file.to_csv(timefile, mode='w', header=False, index=False, encoding = 'utf-8', chunksize=1000)
    else:
        file.to_csv(timefile, mode='a', header=False, index=False, encoding = 'utf-8', chunksize=1000)
    print("..........save_csv", timefile)
    return timefile, time_stamp

def Save_feather(file, filename):  
    cwd = os.getcwd()
    path = os.path.dirname(cwd)

    file_path = path + "\\stock\\data\\"

    #time_stamp = datetime.datetime.now() - datetime.timedelta(hours=12)

    time_stamp = datetime.datetime.now() .strftime('%Y-%m-%d %H:%M:%S')
    timefile = os.path.join(file_path + str(time_stamp[0:11]) +"NQ=F USTime" + filename)
    #file.drop_duplicates(subset='time', keep=False)
    file.to_feather(timefile)
    print("..........save_feather", timefile)
    return timefile, time_stamp

#Running = True  
i = 0  
def get_stock_data():
    start_time = time.time()
    print("*******START TIME(1)", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    price, change, volume = get_price_stock("NQ=F")

    if (i < 800):
        if (float(price) == 0 ):
            print('price, == empyt')
            pass  
        else:
            info = []
            info.append(price)
            info.append(change)
            info.append(volume)
    
            
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
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
    data['price'] = pd.to_numeric(data['price'])
    data['volume'] = pd.to_numeric(data['volume'])
    data.set_index("datetime", inplace=True)
    data_vol=data.copy()
    data= data['price'].resample('1Min').agg({'open': 'first', 
                                 'high': 'max', 
                                 'low': 'min', 
                                 'close': 'last'})

    #d_vol.set_index(['datetime'], inplace=True)
    d_vol = data_vol['volume'].resample('1Min').mean()
    data['volume']=d_vol
    #data = data[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    data.fillna(method="bfill" , inplace=True)
    data.reset_index(inplace=True)
    
    print("***********out_stock_data", data, )
    outputfile, time_stamp = Save_csv(data, '_out_stock_data.csv', 'w' )
    outputfile_feather, time_stamp = Save_feather(data, '_feather_stock_data.feather')
    
    end_time = time.time()
    print("*******END TIME(2)", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))

    return data




       
