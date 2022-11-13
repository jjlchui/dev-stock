import requests 
import datetime
import pandas as pd
from bs4 import BeautifulSoup
import re 
import os
import time



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
            return 0,0,0
        else:
            stock_price = web_content.find("fin-streamer", {'class' : 'Fw(b) Fz(36px) Mb(-4px) D(ib)'}).text
            if(stock_price == ""  ):
                time.sleep(1)
                return 0,0,0
            else:        
                stock_price = stock_price.replace(",", "")
                print('stock price :', stock_price)
                change = web_content.find("fin-streamer", {'data-field' : "regularMarketChangePercent"}).text
                tabl = soup.find_all("div", {'class' : "D(ib) W(1/2) Bxz(bb) Pend(12px) Va(t) ie-7_D(i) smartphone_D(b) smartphone_W(100%) smartphone_Pend(0px) smartphone_BdY smartphone_Bdc($seperatorColor)"})
                change = change.strip('()')
                change = re.sub('%', "", change)
                print('% change :',change)
                web_content1 = soup.find('table', {'class' :'W(100%)'}).text
                words = web_content1.split()
                
                volume = words[1][-11:]
                volume = volume.replace(",", "")
                print("volume: ",volume)
                return stock_price, change, volume  
    except ConnectionError:
        print("Network Issue !!!")
    
    return stock_price, change, volume     

def Save_csv(file, filename):  
    cwd = os.getcwd()
    path = os.path.dirname(cwd)

    #file_path = /home/jjhui/stock_app/plotlydash/NDX.py/
    __file__ = cwd
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, "/jjhui/stock_app/data/")

    time_stamp = datetime.datetime.now() .strftime('%Y-%m-%d %H:%M:%S')
    timefile = os.path.join(file_path + str(time_stamp[0:11]) +"NQ=F USTime" + filename)

    #file.drop_duplicates(subset='time', keep=False)
    file.to_csv(timefile, mode='a', header=False, index=False, encoding = 'utf-8', chunksize=1)
    print("..........save_csv", timefile)
    return timefile, time_stamp

def Process_Data(file):
    #df = pd.read_csv(file, header=None, usecols=[0,1,2,3], names=['datetime', 'price', 'change', 'volume'],
    #                 index_col = ['datetime'], parse_dates=['datetime'])

    data = file.copy()

    #data = pd.DataFrame([data])
    data.columns = ['datetime', 'price', 'change', 'volume']
    print("Hey, Process_Data_test", data.datetime)
    data_time = data.datetime
    
    data['first'] = data['volume'].astype(str).str[0]
    data.drop(data[data["first"] == 'e'].index, inplace=True)
    data = data.drop('first', axis=1)

    data.fillna(method='ffill', inplace=True)
    data.set_index("datetime", inplace=True)
    data.index = pd.DatetimeIndex(data.index)
    
    
    try: 
        #data['price'] = data['price'].astype(float)
        #data['volume'] = data['volume'].astype(float)
        #data['change'] = data['change'].astype(float)
        data['price'] = pd.to_numeric(data['price'])
        data['volume'] = pd.to_numeric(data['volume'])
        data['change'] = pd.to_numeric(data['change'])
    except AttributeError:
        print("cannot conv data to astype float")
        time.sleep(1)
        pass
    
    d_vol = data
    #d_vol.set_index(['datetime'], inplace=True)
    d_vol = d_vol['volume'].resample('1Min').mean()
    d_vol = d_vol.to_frame()
    data = data['price'].resample('1Min').ohlc()
    data['volume'] = d_vol['volume']
    
    data['time'] = data.index
    #data = data.reset_index(drop=True) 
    #data['chk_datetime'] = pd.to_datetime(data_time, format='%Y-%m-%d %H:%M:%S.%f')

    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

    #data = data[['time', 'open', 'high', 'low', 'close', 'chk_datetime']]
    data = data[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    print('head.....',data.head(3))

    #index_with_nan = data.index[data.isnull().any(axis=1)]
    
    #data.drop(index_with_nan, 0, inplace=True)

    data.fillna(method="bfill" , inplace=True)
    data.reset_index(drop=True, inplace=True)
    print("*********data in process", data)
    return data
    


i=0
Running = True
#schedule.every().minutes.do(get_price_stock, "U")

while (Running):
    #schedule.run_pending()
    
    col = []

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if (i < 700):
        price, change, volume = get_price_stock("^NDX")
        print(i)
        print("----------")
        print(" price:", price, " change", change, " volume", volume)
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
            time.sleep(60)  
            
            i = i+1
    else:
        Running = False
    
    if (float(price) == 0):
            pass    
    else:    
        df = pd.DataFrame(col)
        df = df.T
    
    # OUTPUT : _reconcil_stock_data.csv
    
    df.drop_duplicates(keep='first')
    outputfile, time_stamp = Save_csv(df, '_reconcil_stock_data.csv')   

    data = Process_Data(df)    
    
    # OUTPUT : _out_stock_data.csv
    
    data.drop_duplicates(subset=['time'], keep='first')
    outputfile, time_stamp = Save_csv(data, '_out_stock_data.csv')
    
    
    
    """
    cwd = os.getcwd()
    path = os.path.dirname(cwd)
    file_path = path + "\\Data\\"
    timefile = os.path.join(file_path + str(time_stamp[0:11]) +"NDX " + 'stock_data.csv')
    print(timefile)
    df.to_csv(timefile, mode='a', header=False, index=False, encoding = 'utf-8')
    """



