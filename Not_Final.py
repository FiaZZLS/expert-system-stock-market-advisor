

from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import sys
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QPushButton, QWidget ,QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QFileDialog , QSlider , QStylePainter
from PyQt5.QtCore import Qt
from matplotlib import image as mpimg
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
from scipy.stats import linregress
import seaborn as sns
sns.set_style('darkgrid')
import mplfinance as mpf 
import matplotlib.dates as mpdates
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
from scipy.stats import linregress
from progress.bar import Bar
from typing import List, Union

dir_ = os.path.realpath('').split("research")[0]

############################################_HS_##############################################

def pivot_id_HS(ohlc: pd.DataFrame, l:int , n1:int , n2:int ):
    # Check if the length conditions met
    if l-n1 < 0 or l+n2 >= len(ohlc):
        return 0
    
    pivot_low  = 1
    pivot_high = 1

    bar = Bar(f'Processing pivot for n1:{n1} and n2:{n2}', max=len(range(l-n1, l+n2+1)))

    for i in range(l-n1, l+n2+1):
        if(ohlc.loc[l,"Low"] > ohlc.loc[i, "Low"]):
            pivot_low = 0

        if(ohlc.loc[l, "High"] < ohlc.loc[i, "High"]):
            pivot_high = 0


        bar.next()

    bar.finish()
    if pivot_low and pivot_high:
        return 3

    elif pivot_low:
        return 1

    elif pivot_high:
        return 2
    else:
        return 0


def pivot_point_position_HS(row):

    if row['Pivot2']==1:
        return row['Low']-1e-3
    elif row['Pivot2']==2:
        return row['Low']+1e-3
    else:
        return np.nan


def _find_points_HS(df, candle_id, back_candles):
    """
    Find points provides all the necessary arrays and data of interest

    :params df        -> DataFrame with OHLC data
    :params candle_id -> current candle
    :params back_candles -> lookback period
    :return maxim, minim, xxmax, xxmin, maxacount, minacount, maxbcount, minbcount
    """

    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])
    minbcount=0 #minimas before head
    maxbcount=0 #maximas before head
    minacount=0 #minimas after head
    maxacount=0 #maximas after head
    
    for i in range(candle_id-back_candles, candle_id+back_candles):
        if df.loc[i,"ShortPivot2"] == 1:
            minim = np.append(minim, df.loc[i, "Low"])
            xxmin = np.append(xxmin, i)        
            if i < candle_id:
                minbcount=+1
            elif i>candle_id:
                minacount+=1
        if df.loc[i, "ShortPivot2"] == 2:
            maxim = np.append(maxim, df.loc[i, "High"])
            xxmax = np.append(xxmax, i)
            if i < candle_id:
                maxbcount+=1
            elif i>candle_id:
                maxacount+=1
    


    return maxim, minim, xxmax, xxmin, maxacount, minacount, maxbcount, minbcount

def find_inverse_head_and_shoulders(df, back_candles=14):
    """
    Find all the inverse head and shoulders chart patterns

    :params df -> an ohlc dataframe that has "ShortPivot" and "Pivot" as columns
    :params back_candles -> Look-back and look-forward period
    :returns all_points
    """
    all_points = []
    for candle_id in range(back_candles+20, len(df)-back_candles):
        
        if df.loc[candle_id, "Pivot2"] != 1 or df.loc[candle_id,"ShortPivot2"] != 1:
            continue
        

        maxim, minim, xxmax, xxmin, maxacount, minacount, maxbcount, minbcount = _find_points_HS(df, candle_id, back_candles)
        if minbcount<1 or minacount<1 or maxbcount<1 or maxacount<1:
            continue

        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)
        
        headidx = np.argmin(minim, axis=0)
        
        
        
        try:
            if minim[headidx-1]-minim[headidx]>1.5e-3 and minim[headidx+1]-minim[headidx]>1.5e-3 and abs(slmax)<=1e-4: 
                all_points.append(candle_id)
        except:
            pass
            

    return all_points



def find_head_and_shoulders(df: pd.DataFrame, back_candles: int = 14) -> List[int]:

    all_points = []
    for candle_id in range(back_candles+20, len(df)-back_candles):
        
        if df.loc[candle_id, "Pivot2"] != 2 or df.loc[candle_id,"ShortPivot2"] != 2:
            continue
        
        
        maxim, minim, xxmax, xxmin, maxacount, minacount, maxbcount, minbcount = _find_points_HS(df, candle_id, back_candles)
        if minbcount<1 or minacount<1 or maxbcount<1 or maxacount<1:
            continue

        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        headidx = np.argmax(maxim, axis=0)

        
       
        if maxim[headidx]-maxim[headidx-1]>1.5e-3 and maxim[headidx]-maxim[headidx+1]>1.5e-3 and abs(slmin)<=1e-4: 
            all_points.append(candle_id)
            
            

    return all_points



def save_plot_HS(ohlc, all_points, back_candles, fname="head_and_shoulders", hs=True):


    total = len(all_points)
    bar = Bar(f'Processing {fname} images', max=total)

    for j, point in enumerate(all_points):

        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])
        ohlc["HS"] = np.nan

        for i in range(point-back_candles, point+back_candles):
            if ohlc.loc[i,"ShortPivot2"] == 1:
                minim = np.append(minim, ohlc.loc[i, "Low"])
                xxmin = np.append(xxmin, i)        

            if ohlc.loc[i, "ShortPivot2"] == 2:
                maxim = np.append(maxim, ohlc.loc[i, "High"])
                xxmax = np.append(xxmax, i)              

        

        if hs:

            headidx = np.argmax(maxim, axis=0)  

            hsx = ohlc.loc[[xxmax[headidx-1],xxmin[0],xxmax[headidx],xxmin[1],xxmax[headidx+1] ],"Date"]
            hsy = [maxim[headidx-1], minim[0], maxim[headidx], minim[1], maxim[headidx+1]]
        else:

            headidx = np.argmin(minim, axis=0)
            hsx = ohlc.loc[[xxmin[headidx-1],xxmax[0],xxmin[headidx],xxmax[1],xxmin[headidx+1] ],"Date"]
            hsy = [minim[headidx-1], maxim[0], minim[headidx], maxim[1], minim[headidx+1]]

        ohlc_copy = ohlc.copy()
        ohlc_copy.set_index("Date", inplace=True)
        
        levels = [(x,y) for x,y in zip(hsx,hsy)]

        for l in levels:
            ohlc_copy.loc[l[0].strftime("%Y-%m-%dT%H:%M:%S.%f"),"HS"] = l[1]



        ohlc_hs  = ohlc_copy.iloc[point-(back_candles+6):point+back_candles+6, : ]
        hs_l       = mpf.make_addplot(ohlc_hs["HS"], type="scatter", color='r', marker="v", markersize=200)
        fn       = f"{fname}-{point}.png"
        save_   = os.path.join( dir_,"Desktop","Projet system expert",'images','H&S',fn)
        mpf.plot(ohlc_hs,
                type='candle',
                style='charles',
                addplot=[hs_l],
                alines=dict(alines=levels,colors=['purple'], alpha=0.5,linewidths=20),
                savefig=f"{save_}"
                )

        bar.next()
    bar.finish()
    return



############################################_Flag_##############################################

def pivot_id_flag(ohlc, l, n1, n2):

    # Check if the length conditions met
    if l-n1 < 0 or l+n2 >= len(ohlc):
        return 0
    
    pivot_low  = 1
    pivot_high = 1

    for i in range(l-n1, l+n2+1):
        if(ohlc.loc[l,"Low"] > ohlc.loc[i, "Low"]):
            pivot_low = 0

        if(ohlc.loc[l, "High"] < ohlc.loc[i, "High"]):
            pivot_high = 0

    if pivot_low and pivot_high:
        return 3

    elif pivot_low:
        return 1

    elif pivot_high:
        return 2
    else:
        return 0


def pivot_point_position_flag(row):
    """
    Get the Pivot Point position and assign the Low or High value

    :params row -> row of the ohlc dataframe
    :return float
    """
   
    if row['Pivot1']==1:
        return row['Low']-1e-3
    elif row['Pivot1']==2:
        return row['High']+1e-3
    else:
        return np.nan


def find_flag_points(ohlc, back_candles):
    """
    Find flag points

    :params ohlc         -> dataframe that has OHLC data
    :params back_candles -> number of periods to lookback
    :return all_points
    """
    all_points = []
    for candle_idx in range(back_candles+10, len(ohlc)):

        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])

        for i in range(candle_idx-back_candles, candle_idx+1):
            if ohlc.loc[i,"Pivot1"] == 1:
                minim = np.append(minim, ohlc.loc[i, "Low"])
                xxmin = np.append(xxmin, i) 
            if ohlc.loc[i,"Pivot1"] == 2:
                maxim = np.append(maxim, ohlc.loc[i,"High"])
                xxmax = np.append(xxmax, i)

        if (xxmax.size <3 and xxmin.size <3) or xxmax.size==0 or xxmin.size==0:
        
            continue

 
        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)
        if (xxmin[0] < minim[0]):
            pos = "Climbing"
        else:
            pos = "Falling"  
        # Check if the lines are parallel 
        if abs(rmax)>=0.9 and abs(rmin)>=0.9 and (slmin>=1e-3 and slmax>=1e-3 ) or (slmin<=-1e-3 and slmax<=-1e-3):
                        if (slmin/slmax > 0.9 and slmin/slmax < 1.05): # The slopes are almost equal to each other

                            all_points.append(candle_idx)
                            

    return all_points,pos


def save_plot_flag(ohlc, all_points, back_candles , pos):
    """
    Save all the flag graphs

    :params ohlc         -> dataframe that has OHLC data
    :params all_points   -> flag points
    :params back_candles -> number of periods to lookback
    :return 
    """

    total = len(all_points)
    for j, point in enumerate(all_points):

        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])

       
        for i in range(point-back_candles, point+1):
            if ohlc.loc[i,"Pivot1"] == 1:
                minim = np.append(minim, ohlc.loc[i, "Low"])
                xxmin = np.append(xxmin, i) 
                
            if ohlc.loc[i,"Pivot1"] == 2:
                maxim = np.append(maxim, ohlc.loc[i,"High"])
                xxmax = np.append(xxmax, i)

        idx     = range(point-back_candles-5,point-back_candles)
        xslope  = np.array([])
        values  = np.array([])

        for i in idx:
            xslope = np.append(xslope,i)
            values = np.append(values, ohlc.loc[i,"Close"])


        # Linear regressions
        sl, interm, r, p, se = linregress(xslope, values)
        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)

        xxmin = np.append(xxmin, xxmin[-1]) 
        xxmax = np.append(xxmax, xxmax[-1])



        ohlc_subset = ohlc[point-back_candles-5:point+back_candles+5]
        ohlc_subset_copy = ohlc_subset.copy()
        ohlc_subset_copy.loc[:,"Index"] = ohlc_subset_copy.index
        
 

        xxmin = np.append(xxmin, xxmin[-1]+15)
        xxmax = np.append(xxmax, xxmax[-1]+15)


    
        # Make the plot
        fig, ax = plt.subplots(figsize=(15,7))

        
        candlestick_ohlc(ax, ohlc_subset_copy.loc[:, ["Index","Open", "High", "Low", "Close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)

       
        ax.plot(xxmin, xxmin*slmin + intercmin)
        ax.plot(xxmax, xxmax*slmax + intercmax)
        ax.plot(xslope, sl*xslope + interm, color="magenta", linewidth=3)
        ax.annotate(' ', xy=(point,ohlc_subset_copy.loc[point,"High"]), arrowprops=dict(width=9, headlength = 14, headwidth=14, facecolor='purple', color='purple') )

        ax.grid(True)
        ax.set_xlabel('Index')
        ax.set_ylabel('Price')

        
        if sl < 0 and slmin > 0 and slmax > 0:
            name= f"potential-bearish-{point}"
        elif sl >0 and slmin <0 and slmax <0:
            name= f"potential-bullish-{point}"
        else:
            name = f"{point}"
        if (name != str(point)):
            fn = f"flag-{name}.png"
            file = os.path.join( dir_,"Desktop","Projet system expert",'images','Flag',fn)
            plt.savefig(file, format="png")
            print(f"Completed {round((j+1)/total,2)*100}%")

    return
############################################_Wedge_##############################################

def pivot_id_wedge(ohlc, l, n1, n2):
    # Check if the length conditions met
    if l-n1 < 0 or l+n2 >= len(ohlc):
        return 0
    
    pivot_low  = 1
    pivot_high = 1

    for i in range(l-n1, l+n2+1):
        if(ohlc.loc[l,"Close"] > ohlc.loc[i, "Close"]):
            pivot_low = 0

        if(ohlc.loc[l, "Close"] < ohlc.loc[i, "Close"]):
            pivot_high = 0

    if pivot_low and pivot_high:
        return 3

    elif pivot_low:
        return 1

    elif pivot_high:
        return 2
    else:
        return 0


def pivot_point_position_wedge(row):
    """
    Get the Pivot Point position and assign a Close value

    :params row -> row of the ohlc dataframe
    :return float
    """
   
    if row['Pivot']==1:
        return row['Close']-1e-3
    elif row['Pivot']==2:
        return row['Close']+1e-3
    else:
        return np.nan


def find_wedge_points(ohlc, back_candles):
    """
    Find wedge points

    :params ohlc         -> dataframe that has OHLC data
    :params back_candles -> number of periods to lookback
    :return all_points
    """
    all_points = []
    for candle_idx in range(back_candles+10, len(ohlc)):

        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])

        for i in range(candle_idx-back_candles, candle_idx+1):
            if ohlc.loc[i,"Pivot"] == 1:
                minim = np.append(minim, ohlc.loc[i, "Close"])
                xxmin = np.append(xxmin, i) 
            if ohlc.loc[i,"Pivot"] == 2:
                maxim = np.append(maxim, ohlc.loc[i,"Close"])
                xxmax = np.append(xxmax, i)

        
        if (xxmax.size <3 and xxmin.size <3) or xxmax.size==0 or xxmin.size==0:
            continue

        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)
        

        # Check if the lines are in the same direction
        if abs(rmax)>=0.9 and abs(rmin)>=0.9 and ((slmin>=1e-3 and slmax>=1e-3 ) or (slmin<=-1e-3 and slmax<=-1e-3)):
                # Check if lines are parallel but converge fast 
                x_ =   (intercmin -intercmax)/(slmax-slmin)
                cors = np.hstack([xxmax, xxmin])  
                if (x_ - max(cors))>0 and (x_ - max(cors))<(max(cors) - min(cors))*3 and slmin/slmax > 0.75 and slmin/slmax < 1.25:  
                     all_points.append(candle_idx)
            

    return all_points


def point_position_plot_wedge(ohlc, start_index, end_index):
        """
        Plot the pivot points over a sample period

        :params ohlc        -> dataframe that has OHLC data
        :params start_index -> index where to start taking the sample data
        :params end_index   -> index where to stop taking the sample data
        :return 
        """
        ohlc_subset = ohlc[start_index:end_index]
        ohlc_subset_copy = ohlc_subset.copy()
        ohlc_subset_copy.loc[:,"Index"] = ohlc_subset_copy.index 



        fig, ax = plt.subplots(figsize=(15,7))
        candlestick_ohlc(ax, ohlc_subset_copy.loc[:, ["Index","Open", "High", "Low", "Close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
        ax.scatter(ohlc_subset_copy["Index"], ohlc_subset_copy["PointPos"])

        ax.grid(True)
        ax.set_xlabel('Index')
        ax.set_ylabel('Price')


        fn   = f"wedge-pivot-point-sample.png"
        file = os.path.join( dir_,"Desktop","Projet system expert",'images','Wedge',fn)
        plt.savefig(file, format="png")

        return

def save_plot_wedge(ohlc, all_points, back_candles):
    """
    Save all the wedge graphs

    :params ohlc         -> dataframe that has OHLC data
    :params all_points   -> wedge points
    :params back_candles -> number of periods to lookback
    :return 
    """

    total = len(all_points)
    for j, point in enumerate(all_points):

        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])

        for i in range(point-back_candles, point+1):
            if ohlc.loc[i,"Pivot"] == 1:
                minim = np.append(minim, ohlc.loc[i, "Close"])
                xxmin = np.append(xxmin, i) 
            if ohlc.loc[i,"Pivot"] == 2:
                maxim = np.append(maxim, ohlc.loc[i,"Close"])
                xxmax = np.append(xxmax, i)
                

        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)

        xxmin = np.append(xxmin, xxmin[-1]) 
        xxmax = np.append(xxmax, xxmax[-1])

        ohlc_subset = ohlc[point-back_candles-5:point+back_candles+5]
        ohlc_subset_copy = ohlc_subset.copy()
        ohlc_subset_copy.loc[:,"Index"] = ohlc_subset_copy.index
    
        xxmin = np.append(xxmin, xxmin[-1]+15)
        xxmax = np.append(xxmax, xxmax[-1]+15)

        fig, ax = plt.subplots(figsize=(15,7))

        
        candlestick_ohlc(ax, ohlc_subset_copy.loc[:, ["Index","Open", "High", "Low", "Close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
        ax.plot(xxmin, xxmin*slmin + intercmin)
        ax.plot(xxmax, xxmax*slmax + intercmax)

        ax.grid(True)
        ax.set_xlabel('Index')
        ax.set_ylabel('Price')

        current_index = point # Change this to your desired index
        next_2H = ohlc['High'].shift(-1).iloc[current_index:current_index+3].values
        previous_2H = ohlc['High'].shift(1).iloc[current_index-14:current_index].values
        next_2O = ohlc['Open'].shift(-1).iloc[current_index:current_index+3].values
        previous_2O = ohlc['Open'].shift(1).iloc[current_index-14:current_index].values
        next_2C = ohlc['Close'].shift(-1).iloc[current_index:current_index+3].values
        previous_2C = ohlc['Close'].shift(1).iloc[current_index-14:current_index].values
        next_2L = ohlc['Low'].shift(-1).iloc[current_index:current_index+3].values
        previous_2L = ohlc['Low'].shift(1).iloc[current_index-14:current_index].values
        next_F = (next_2H[0] + next_2O[0] + next_2C[0] + next_2L[0])/4
        previous_F = (previous_2H[0] + previous_2O[0] + previous_2C[0] + previous_2L[0])/4
        print(next_F)
        print("\nHigh values in the previous 2 indexes:")
        print(previous_F)
        if (previous_F > next_F):
            name2= "Falling"
        else:
            name2 = "Climbing"
        fn = f"wedge-{point}-{name2}.png"
        file = os.path.join( dir_,"Desktop","Projet system expert",'images','Wedge',fn)
        plt.savefig(file, format="png") 
        print(f"Completed {round((j+1)/total,2)*100}%")

    return

####################################################_DOUBLE_##########################################################

def save_plots_double(ohlc, patterns, max_min, filename):
    """
    Save the plots of the analysis

    :params ohlc -> dataframe holding the ohlc data

    :params paterns -> all the indices where the patterns exist

    :params max_min -> the maximas and minimas

    :params filename -> prefix for the graph names 
    """
    for i, pattern in enumerate(patterns):
        fig, ax  = plt.subplots(figsize=(15,7)) 
        start_   = pattern[0]
        end_     = pattern[1]
        idx      = max_min.loc[start_-100:end_+100].index.values.tolist()
        ohlc_copy = ohlc.copy()
        ohlc_copy.loc[:,"Index"] = ohlc_copy.index

        max_min_idx = max_min.loc[start_:end_].index.tolist()

        candlestick_ohlc(ax, ohlc_copy.loc[idx, ["Index","Open", "High", "Low", "Close"] ].values, width=0.1, colorup='green', colordown='red', alpha=0.8)   
        ax.plot(max_min_idx, max_min.loc[start_:end_].values[:,1],color='orange')


        ax.grid(True)
        ax.set_xlabel('Index')
        ax.set_ylabel('Price')

        # Create the save folder if it does not exist


        fn = f"{filename}-{i}.png"
        file = os.path.join( dir_,"Desktop","Projet system expert",'images','Double',fn)
        plt.savefig(file, format="png")

        
        print(f"Completed {round((i+1)/len(patterns),2)*100}%")

    return 




def find_doubles_patterns(max_min):
    """
    Find the the double tops and bottoms patterns

    :params max_min -> the maximas and minimas

    :return patterns_tops, patterns_bottoms
    """

    # Placeholders for the Double Tops and Bottoms indices
    patterns_tops = []
    patterns_bottoms = []

    # Window range is 5 units
    for i in range(5, len(max_min)):
        window = max_min.iloc[i-5:i]


        # Pattern must play out in less than n units
        if window.index[-1] - window.index[0] > 50:
            continue

        a, b, c, d, e = window.iloc[0:5,1]
        # Double Tops
        if a<b and a<d and c<b and c<d and e<b and e<d and b>d :
               patterns_tops.append((window.index[0], window.index[-1]))

        # Double Bottoms
        if a>b and a>d and c>b and c>d and e>b and e>d and b<d:
                patterns_bottoms.append((window.index[0], window.index[-1]))

    return patterns_tops, patterns_bottoms


def find_local_maximas_minimas_double(ohlc, window_range, smooth=False, smoothing_period=10):
    """
    Find all the local maximas and minimas

    :params ohlc         -> dataframe holding the ohlc data
    :params window_range -> range to find min and max
    :params smooth       -> should the prices be smoothed
    :params smoothing_period -> the smoothing period

    :return max_min
    """
    local_max_arr = []
    local_min_arr = []

    if smooth:
        smooth_close = ohlc["Close"].rolling(window=smoothing_period).mean().dropna()
        local_max = argrelextrema(smooth_close.values, np.greater)[0]
        local_min = argrelextrema(smooth_close.values, np.less)[0]
    else:
        local_max = argrelextrema(ohlc["Close"].values, np.greater)[0]
        local_min = argrelextrema(ohlc["Close"].values, np.less)[0]

    for i in local_max:
        if (i>window_range) and (i<len(ohlc)-window_range):
            local_max_arr.append(ohlc.iloc[i-window_range:i+window_range]['Close'].idxmax())

    for i in local_min:
        if (i>window_range) and (i<len(ohlc)-window_range):
            local_min_arr.append(ohlc.iloc[i-window_range:i+window_range]['Close'].idxmin())


    maxima  = pd.DataFrame(ohlc.loc[local_max_arr])
    minima  = pd.DataFrame(ohlc.loc[local_min_arr])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min = max_min[~max_min.index.duplicated()]

    return max_min

###########################################################EL INTERFACE FINAL###################################################################

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Market Advisor")
        self.setGeometry(100, 100, 800, 600)


        # Set the background image
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.central_widget = InputPage(self)
        self.setCentralWidget(self.central_widget)

class InputPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        pic_path = os.path.join( dir_,"Desktop","Projet system expert","BG_image.JPG")

        input_layout = QHBoxLayout()

        self.start_date_label = QLabel("Start Date:")
        self.start_date_input = QLineEdit(self)
        self.start_date_label.setStyleSheet("font-size: 18px;")
        input_layout.addWidget(self.start_date_label)
        input_layout.addWidget(self.start_date_input)

        self.end_date_label = QLabel("End Date:")
        self.end_date_input = QLineEdit(self)
        self.end_date_label.setStyleSheet("font-size: 18px;")
        input_layout.addWidget(self.end_date_label)
        input_layout.addWidget(self.end_date_input)

        layout.addLayout(input_layout)

        image_label = QLabel(self)
        pixmap = QPixmap(pic_path)
        pixmap = pixmap.scaled(900, 500)  # Resize the image
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)

        start_button = QPushButton("Start")
        start_button.setStyleSheet("font-size: 18px;")
        start_button.clicked.connect(self.start_analysis)
        layout.addWidget(start_button)

        open_folder_button = QPushButton("Open Folder")
        open_folder_button.setStyleSheet("font-size: 18px;")
        open_folder_button.clicked.connect(self.open_folder)
        layout.addWidget(open_folder_button)

        self.setLayout(layout)
    def start_analysis(self):
        # Get the input values
        start_date = self.start_date_input.text()
        end_date = self.end_date_input.text()

        # You can use start_date and end_date for further processing
        print("Start Date:", start_date)
        print("End Date:", end_date)


###########################################_Flag_##############################################
        ohlc["Pivot1"] = 0


        # # Get the minimas and maximas 
        ohlc["Pivot1"]    = ohlc.apply(lambda x: pivot_id_flag(ohlc, x.name, 3, 3), axis=1)
        ohlc['PointPos1'] = ohlc.apply(lambda row: pivot_point_position_flag(row), axis=1)

        # Find all flag pattern points
        back_candles = 20
        all_points , pos = find_flag_points(ohlc, back_candles)

        # Plot the flag pattern graphs
        save_plot_flag(ohlc, all_points, back_candles , pos)
# ############################################_Wedge_##############################################
        ohlc = ohlc[real_columns]
        ohlc["Pivot"] = 0


        # Get the minimas and maximas 
        ohlc["Pivot"]    = ohlc.apply(lambda x: pivot_id_wedge(ohlc, x.name, 3, 3), axis=1)
        ohlc['PointPos'] = ohlc.apply(lambda x: pivot_point_position_wedge(x), axis=1) # Used for visualising the pivot points


        # Plot sample point positions
        point_position_plot_wedge(ohlc, 50, 200)

        # # Find all wedge pattern points
        back_candles = 20
        all_points   = find_wedge_points(ohlc, back_candles)

        # Plot the wedge pattern graphs
        save_plot_wedge(ohlc, all_points, back_candles)

############################################_Double_##############################################
        ohlc = ohlc[real_columns]
        # Find all the local minimas and maximas
        window_range = 10 # Defining the local range where min and max will be found
        max_min = find_local_maximas_minimas_double(ohlc, window_range, smooth=True)

        # Find the tops and bottoms
        patterns_tops, patterns_bottoms = find_doubles_patterns(max_min)

        # Plots for Double Tops
        print("Plotting the Double Tops")
        save_plots_double(ohlc, patterns_tops, max_min, "double-tops")
        

        # Plots for Double Bottoms
        print("Plotting the Double Bottoms")
        save_plots_double(ohlc, patterns_bottoms, max_min, "double-bottoms")
# ############################################_H&S_##############################################
#         ohlc = df.loc[:, ["Date", "Open", "High", "Low", "Close"] ]
#         ohlc["Date"] = pd.DatetimeIndex(ohlc["Date"]) 
#         ohlc["Pivot2"] = ohlc.apply(lambda x: pivot_id_HS(ohlc, x.name, 15, 15), axis=1)
#         ohlc['ShortPivot2'] = ohlc.apply(lambda x: pivot_id_HS(df, x.name,5,5), axis=1)
#         ohlc['PointPos2'] = ohlc.apply(lambda row: pivot_point_position_HS(row), axis=1)
    
#         back_candles =14
#         # all_points         = find_head_and_shoulders(ohlc,back_candles=back_candles)
#         all_points_inverse = find_inverse_head_and_shoulders(ohlc, back_candles=back_candles)
        
#         # Save plots
#         # save_plot_HS(ohlc, all_points, back_candles)
#         save_plot_HS(ohlc, all_points_inverse, back_candles, fname="inverse_head_and_shoulders", hs=False)
    def open_folder(self):
        folder_paths =[]
        folder_paths.append(os.path.join( dir_,"Desktop","Projet system expert",'images','Double'))
        folder_paths.append(os.path.join( dir_,"Desktop","Projet system expert",'images','Flag'))
        folder_paths.append(os.path.join( dir_,"Desktop","Projet system expert",'images','H&S'))
        folder_paths.append(os.path.join( dir_,"Desktop","Projet system expert",'images','Wedge'))
        finalImages = []
        decisions =[]
        for folder_path in folder_paths:
            if folder_path:
                # List image files in the selected folder
                self.image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                for i in self.image_paths:
                    finalImages.append(i)
                # Create a new interface to display the images
            
                    if ("tops" in i):
                        decisions.append("Buy this stock")
                    if ("bottoms" in i):
                        decisions.append("Sell this stock")   
                    if ("bearish" in i):
                        decisions.append("Sell this stock")  
                    if ("bullish" in i):
                        decisions.append("Buy this stock") 
                    if ("Falling" in i) :
                        decisions.append("Buy this stock") 
                    if ("Climbing" in i) :
                        decisions.append("Sell this stock")

                    
                self.parent.setCentralWidget(self.parent.central_widget)
        print(decisions)
        self.parent.central_widget = ImageDisplayPage(self.parent, finalImages, decisions)
class ImageDisplayPage(QWidget):
    def __init__(self, parent, image_paths, image_labels):
        super().__init__(parent)
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.current_image_index = 0

        layout = QVBoxLayout()

        self.image_label = QLabel()

        self.update_image()
        self.label = QLabel(self.image_labels[self.current_image_index])
        self.label.setStyleSheet("font-size: 40px;")

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(len(image_paths) - 1)
        slider.setValue(0)
        slider.valueChanged.connect(self.slider_value_changed)

        layout.addWidget(self.image_label)
        layout.addWidget(self.label)
        layout.addWidget(slider)

        self.setLayout(layout)

    def slider_value_changed(self, value):
        self.current_image_index = value
        self.update_image()
        self.update_label()

    def update_image(self):
        if 0 <= self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)

    def update_label(self):
        self.label.setText(self.image_labels[self.current_image_index])
def main():
    directory_paths = []
    directory_paths.append(os.path.join(dir_, "Desktop", "Projet system expert", 'images', 'Double'))
    directory_paths.append(os.path.join(dir_, "Desktop", "Projet system expert", 'images', 'Wedge'))
    directory_paths.append(os.path.join(dir_, "Desktop", "Projet system expert", 'images', 'Flag'))
    for directory_path in directory_paths:
        contents = os.listdir(directory_path)
        for item in contents:
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                os.rmdir(item_path)
    app = QApplication(sys.argv)
    window = MainMenu()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
