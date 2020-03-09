## Weather prediction using GRU architecture

## Dataset
#### Unprocessed dataset 
It was taken from official NCDC website:  
https://www7.ncdc.noaa.gov/CDO/country  
**You could follow those steps to obtain dataset file by yourself:**
- Choose **United States**, then click **Access Data/Products**
- Choose **Surface Data, Hourly Global**
- Choose **Continue With SIMPLIFIED options**
- Select **Retieve data for: New York**
- Select **Country: United States**
- Then find and select required weather station:  
  **Selected UNITED STATES stations: EAST HAMPTON AIRPORT.................... 72209864761 01/2006 to 03/2020**  
  <img src="https://raw.githubusercontent.com/dredwardhyde/gru-weather-prediction/master/dataset/station.PNG" width="700"/>
- Then select required period:  
  <img src="https://raw.githubusercontent.com/dredwardhyde/gru-weather-prediction/master/dataset/period.PNG" width="700"/>
- Click continue and follow instructions how to download requested data
- Unprocessed data looks like that:
```
  USAF  WBAN YR--MODAHRMN DIR SPD GUS CLG SKC L M H  VSB MW MW MW MW AW AW AW AW W TEMP DEWP    SLP   ALT    STP MAX MIN PCP01 PCP06 PCP24 PCPXX SD
722098 64761 201701010255 990  13  23  85 *** * * * 10.0 ** ** ** ** ** ** ** ** *   46   33 ****** 29.93 ****** *** *** ***** ***** ***** ***** ** 
722098 64761 201701010315 240  16  25  90 *** * * * 10.0 ** ** ** ** ** ** ** ** *   46   32 ****** 29.93 ****** *** *** ***** ***** ***** ***** ** 
722098 64761 201701010335 240  15  25  70 *** * * * 10.0 ** ** ** ** ** ** ** ** *   46   31 ****** 29.93 ****** *** *** ***** ***** ***** ***** ** 
722098 64761 201701010355 240  16  26  50 *** * * * 10.0 ** ** ** ** ** ** ** ** *   46   31 ****** 29.94 ****** *** *** ***** ***** ***** ***** ** 
722098 64761 201701010415 250  14  29  50 *** * * * 10.0 ** ** ** ** 51 ** ** ** *   44   31 ****** 29.94 ****** *** *** ***** ***** ***** ***** ** 
722098 64761 201701010435 240  13  23  48 *** * * * 10.0 ** ** ** ** ** ** ** ** *   45   32 ****** 29.92 ****** *** *** ***** ***** ***** ***** ** 
722098 64761 201701010455 230  16  23 100 *** * * * 10.0 ** ** ** ** ** ** ** ** *   45   32 ****** 29.91 1010.8  47  27 ***** ***** ***** ***** ** 
```

#### Processed dataset
```
                     temp  (Various, Day)  (Various, Hour)  (Various, Minute)
time                                                                         
2017-01-01 02:55:00    46               1                2                 55
2017-01-01 03:15:00    46               1                3                 15
2017-01-01 03:35:00    46               1                3                 35
2017-01-01 03:55:00    46               1                3                 55
2017-01-01 04:15:00    44               1                4                 15

Training data:
x_data:  (62455, 4)
y_data:  (62455, 1)
Test data:
x_data:  (15614, 4)
y_data:  (15614, 1)
```

## Architecture
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
gru (GRU)                    (None, None, 128)         51456     
_________________________________________________________________
gru_1 (GRU)                  (None, None, 128)         99072     
_________________________________________________________________
time_distributed (TimeDistri (None, None, 100)         12900     
_________________________________________________________________
time_distributed_1 (TimeDist (None, None, 1)           101       
=================================================================
Total params: 163,529
Trainable params: 163,529
Non-trainable params: 0
_________________________________________________________________
```

## Results

On training data:  
<img src="https://raw.githubusercontent.com/dredwardhyde/gru-weather-prediction/master/results/temp_30000_train.png" width="900"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/gru-weather-prediction/master/results/temp_40000_train.png" width="900"/>  

On test data:  
<img src="https://raw.githubusercontent.com/dredwardhyde/gru-weather-prediction/master/results/temp_4000_test.png" width="900"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/gru-weather-prediction/master/results/temp_5000_test.png" width="900"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/gru-weather-prediction/master/results/temp_6000_test.png" width="900"/>  
