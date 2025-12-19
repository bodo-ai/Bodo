# Series

Bodo provides extensive Series support. However, operations between
Series (+, -, /, *, **) do not implicitly align values based on their
associated index values yet.


### Attributes


- [`pd.Series`][pdseries]                                                
- [`pd.Series.index`][pdseriesindex]                                     
- [`pd.Series.values`][pdseriesvalues]                                   
- [`pd.Series.dtype`][pdseriesdtype]                                     
- [`pd.Series.shape`][pdseriesshape]                                     
- [`pd.Series.nbytes`][pdseriesnbytes]                                   
- [`pd.Series.ndim`][pdseriesndim]                                       
- [`pd.Series.size`][pdseriessize]                                       
- [`pd.Series.T`][pdseriest]                                             
- [`pd.Series.memory_usage`][pdseriesmemory_usage]                       
- [`pd.Series.hasnans`][pdserieshasnans]                                 
- [`pd.Series.empty`][pdseriesempty]                                     
- [`pd.Series.dtypes`][pdseriesdtypes]                                   
- [`pd.Series.name`][pdseriesname]                                       

### Conversion


- [`pd.Series.astype`][pdseriesastype]        
- [`pd.Series.copy`][pdseriescopy]            
- [`pd.Series.to_numpy`][pdseriesto_numpy]    
- [`pd.Series.tolist`][pdseriestolist]        

### Indexing, iteration

Location based indexing using `[]`, `iat`, and
`iloc` is supported. Changing values of existing string
Series using these operators is not supported yet.

- [`pd.Series.iat`][pdseriesiat]                                       
- [`pd.Series.iloc`][pdseriesiloc]                                     
- [`pd.Series.loc`][pdseriesloc]                                       

### Binary operator functions

- [`pd.Series.add`][pdseriesadd]                                        
- [`pd.Series.sub`][pdseriessub]                                        
- [`pd.Series.mul`][pdseriesmul]                                        
- [`pd.Series.div`][pdseriesdiv]                                        
- [`pd.Series.truediv`][pdseriestruediv]                                
- [`pd.Series.floordiv`][pdseriesfloordiv]                              
- [`pd.Series.mod`][pdseriesmod]                                        
- [`pd.Series.pow`][pdseriespow]                                        
- [`pd.Series.radd`][pdseriesradd]                                      
- [`pd.Series.rsub`][pdseriesrsub]                                      
- [`pd.Series.rmul`][pdseriesrmul]                                      
- [`pd.Series.rdiv`][pdseriesrdiv]                                      
- [`pd.Series.rtruediv`][pdseriesrtruediv]                              
- [`pd.Series.rfloordiv`][pdseriesrfloordiv]                            
- [`pd.Series.rmod`][pdseriesrmod]                                      
- [`pd.Series.rpow`][pdseriesrpow]                                      
- [`pd.Series.combine`][pdseriescombine]                                
- [`pd.Series.round`][pdseriesround]                                    
- [`pd.Series.lt`][pdserieslt]                                          
- [`pd.Series.gt`][pdseriesgt]                                          
- [`pd.Series.le`][pdseriesle]                                          
- [`pd.Series.ge`][pdseriesge]                                          
- [`pd.Series.ne`][pdseriesne]                                          
- [`pd.Series.eq`][pdserieseq]                                          
- [`pd.Series.dot`][pdseriesdot]                                        

### Function application, GroupBy & Window


- [`pd.Series.apply`][pdseriesapply]    
- [`pd.Series.map`][pdseriesmap]        
- [`pd.Series.groupby`][pdseriesgroupby]
- [`pd.Series.rolling`][pdseriesrolling]
- [`pd.Series.pipe`][pdseriespipe]      

### Computations / Descriptive Stats

Statistical functions below are supported without optional arguments
unless support is explicitly mentioned.

- [`pd.Series.abs`][pdseriesabs]                                         
- [`pd.Series.all`][pdseriesall]                                         
- [`pd.Series.any`][pdseriesany]                                         
- [`pd.Series.autocorr`][pdseriesautocorr]                               
- [`pd.Series.between`][pdseriesbetween]                                 
- [`pd.Series.corr`][pdseriescorr]                                       
- [`pd.Series.count`][pdseriescount]                                     
- [`pd.Series.cov`][pdseriescov]                                         
- [`pd.Series.cummin`][pdseriescummin]                                   
- [`pd.Series.cummax`][pdseriescummax]                                   
- [`pd.Series.cumprod`][pdseriescumprod]                                 
- [`pd.Series.cumsum`][pdseriescumsum]                                   
- [`pd.Series.describe`][pdseriesdescribe]                               
- [`pd.Series.diff`][pdseriesdiff]                                       
- [`pd.Series.kurt`][pdserieskurt]                                       
- [`pd.Series.max`][pdseriesmax]                                         
- [`pd.Series.mean`][pdseriesmean]                                       
- [`pd.Series.median`][pdseriesmedian]                                   
- [`pd.Series.min`][pdseriesmin]                                         
- [`pd.Series.nlargest`][pdseriesnlargest]                               
- [`pd.Series.nsmallest`][pdseriesnsmallest]                             
- [`pd.Series.pct_change`][pdseriespct_change]                           
- [`pd.Series.prod`][pdseriesprod]                                       
- [`pd.Series.product`][pdseriesproduct]                                 
- [`pd.Series.quantile`][pdseriesquantile]                               
- [`pd.Series.rank`][pdseriesrank]                                       
- [`pd.Series.sem`][pdseriessem]                                         
- [`pd.Series.skew`][pdseriesskew]                                       
- [`pd.Series.std`][pdseriesstd]                                         
- [`pd.Series.sum`][pdseriessum]                                         
- [`pd.Series.var`][pdseriesvar]                                         
- [`pd.Series.kurtosis`][pdserieskurtosis]                               
- [`pd.Series.unique`][pdseriesunique]                                   
- [`pd.Series.nunique`][pdseriesnunique]                                 
- [`pd.Series.is_monotonic_increasing`][pdseriesis_monotonic_increasing] 
- [`pd.Series.is_monotonic_decreasing`][pdseriesis_monotonic_decreasing] 
- [`pd.Series.value_counts`][pdseriesvalue_counts]                       

### Reindexing / Selection / Label manipulation

- [`pd.Series.drop_duplicates`][pdseriesdrop_duplicates]                 
- [`pd.Series.duplicated`][pdseriesduplicated]                           
- [`pd.Series.equals`][pdseriesequals]                                   
- [`pd.Series.first`][pdseriesfirst]                                     
- [`pd.Series.head`][pdserieshead]                                       
- [`pd.Series.idxmax`][pdseriesidxmax]                                   
- [`pd.Series.idxmin`][pdseriesidxmin]                                   
- [`pd.Series.isin`][pdseriesisin]                                       
- [`pd.Series.last`][pdserieslast]                                       
- [`pd.Series.rename`][pdseriesrename]                                   
- [`pd.Series.reset_index`][pdseriesreset_index]                         
- [`pd.Series.take`][pdseriestake]                                       
- [`pd.Series.tail`][pdseriestail]                                       
- [`pd.Series.where`][pdserieswhere]                                     
- [`pd.Series.mask`][pdseriesmask]                                       

### Missing data handling

- [`pd.Series.backfill`][pdseriesbackfill]                               
- [`pd.Series.bfill`][pdseriesbfill]                                     
- [`pd.Series.dropna`][pdseriesdropna]                                   
- [`pd.Series.ffill`][pdseriesffill]                                     
- [`pd.Series.fillna`][pdseriesfillna]                                   
- [`pd.Series.isna`][pdseriesisna]                                       
- [`pd.Series.isnull`][pdseriesisnull]                                   
- [`pd.Series.notna`][pdseriesnotna]                                     
- [`pd.Series.notnull`][pdseriesnotnull]                                 
- [`pd.Series.pad`][pdseriespad]                                         
- [`pd.Series.replace`][pdseriesreplace]                                 

### Reshaping, sorting


- [`pd.Series.argsort`][pdseriesargsort]                                 
- [`pd.Series.sort_values`][pdseriessort_values]                         
- [`pd.Series.sort_index`][pdseriessort_index]                           
- [`pd.Series.explode`][pdseriesexplode]                                 
- [`pd.Series.repeat`][pdseriesrepeat]                                   
                               

### Time series-related


- [`pd.Series.shift`][pdseriesshift]                                     


### Datetime properties


- [`pd.Series.dt.date`][pdseriesdtdate]                                   
- [`pd.Series.dt.year`][pdseriesdtyear]                                  
- [`pd.Series.dt.month`][pdseriesdtmonth]                                
- [`pd.Series.dt.day`][pdseriesdtday]                                    
- [`pd.Series.dt.hour`][pdseriesdthour]                                  
- [`pd.Series.dt.minute`][pdseriesdtminute]                              
- [`pd.Series.dt.second`][pdseriesdtsecond]                              
- [`pd.Series.dt.microsecond`][pdseriesdtmicrosecond]                    
- [`pd.Series.dt.nanosecond`][pdseriesdtnanosecond]                      
- [`pd.Series.dt.day_of_week`][pdseriesdtday_of_week]                    
- [`pd.Series.dt.weekday`][pdseriesdtweekday]                            
- [`pd.Series.dt.dayofyear`][pdseriesdtdayofyear]                        
- [`pd.Series.dt.day_of_year`][pdseriesdtday_of_year]                    
- [`pd.Series.dt.quarter`][pdseriesdtquarter]                            
- [`pd.Series.dt.is_month_start`][pdseriesdtis_month_start]              
- [`pd.Series.dt.is_month_end`][pdseriesdtis_month_end]                  
- [`pd.Series.dt.is_quarter_start`][pdseriesdtis_quarter_start]          
- [`pd.Series.dt.is_quarter_end`][pdseriesdtis_quarter_end]              
- [`pd.Series.dt.is_year_start`][pdseriesdtis_year_start]                
- [`pd.Series.dt.is_year_end`][pdseriesdtis_year_end]                    
- [`pd.Series.dt.is_leap_year`][pdseriesdtis_leap_year]                    
- [`pd.Series.dt.daysinmonth`][pdseriesdtdaysinmonth]                    
- [`pd.Series.dt.days_in_month`][pdseriesdtdays_in_month]                

### Datetime methods


- [`pd.Series.dt.normalize`][pdseriesdtnormalize]                        
- [`pd.Series.dt.strftime`][pdseriesdtstrftime]                          
- [`pd.Series.dt.round`][pdseriesdtround]                                
- [`pd.Series.dt.floor`][pdseriesdtfloor]                                
- [`pd.Series.dt.ceil`][pdseriesdtceil]                                  
- [`pd.Series.dt.month_name`][pdseriesdtmonth_name]                      
- [`pd.Series.dt.day_name`][pdseriesdtday_name]                          

### String handling


- [`pd.Series.str.capitalize`][pdseriesstrcapitalize]                    
- [`pd.Series.str.cat`][pdseriesstrcat]                                  
- [`pd.Series.str.center`][pdseriesstrcenter]                            
- [`pd.Series.str.contains`][pdseriesstrcontains]                        
- [`pd.Series.str.count`][pdseriesstrcount]                              
- [`pd.Series.str.endswith`][pdseriesstrendswith]                        
- [`pd.Series.str.extract`][pdseriesstrextract]                          
- [`pd.Series.str.extractall`][pdseriesstrextractall]                    
- [`pd.Series.str.find`][pdseriesstrfind]                                
- [`pd.Series.str.get`][pdseriesstrget]                                  
- [`pd.Series.str.join`][pdseriesstrjoin]                                
- [`pd.Series.str.len`][pdseriesstrlen]                                  
- [`pd.Series.str.ljust`][pdseriesstrljust]                              
- [`pd.Series.str.lower`][pdseriesstrlower]                              
- [`pd.Series.str.lstrip`][pdseriesstrlstrip]                            
- [`pd.Series.str.pad`][pdseriesstrpad]                                  
- [`pd.Series.str.repeat`][pdseriesstrrepeat]                            
- [`pd.Series.str.replace`][pdseriesstrreplace]                          
- [`pd.Series.str.rfind`][pdseriesstrrfind]                              
- [`pd.Series.str.rjist`][pdseriesstrrjist]                              
- [`pd.Series.str.restrip`][pdseriesstrrestrip]                          
- [`pd.Series.str.slice`][pdseriesstrslice]                              
- [`pd.Series.str.slice_replace`][pdseriesstrslice_replace]              
- [`pd.Series.str.split`][pdseriesstrsplit]                              
- [`pd.Series.str.startswith`][pdseriesstrstartswith]                    
- [`pd.Series.str.strip`][pdseriesstrstrip]                              
- [`pd.Series.str.swapcase`][pdseriesstrswapcase]                        
- [`pd.Series.str.title`][pdseriesstrtitle]                              
- [`pd.Series.str.upper`][pdseriesstrupper]                              
- [`pd.Series.str.zfill`][pdseriesstrzfill]                              
- [`pd.Series.str.isalnum`][pdseriesstrisalnum]                          
- [`pd.Series.str.isalpha`][pdseriesstrisalpha]                          
- [`pd.Series.str.isdigit`][pdseriesstrisdigit]                          
- [`pd.Series.str.isspace`][pdseriesstrisspace]                          
- [`pd.Series.str.islower`][pdseriesstrislower]                          
- [`pd.Series.str.isupper`][pdseriesstrisupper]                          
- [`pd.Series.str.istitle`][pdseriesstristitle]                          
- [`pd.Series.str.isnumeric`][pdseriesstrisnumeric]                      
- [`pd.Series.str.isdecimal`][pdseriesstrisdecimal]   
- [`pd.Series.str.encode`][pdseriesstrencode]                   

### Categorical accessor


- [`pd.Series.cat.codes`][pdseriescatcodes]                              

### Serialization / IO / Conversion

- [`pd.Series.to_csv`][pdseriesto_csv]                                   
- [`pd.Series.to_dict`][pdseriesto_dict]                                 
- [`pd.Series.to_frame`][pdseriesto_frame]                               

## Heterogeneous Series {#heterogeneous_series}

Bodo's Series implementation requires all elements to share a common
data type. However, in situations where the size and types of the
elements are constant at compile time, Bodo has some mixed type handling
with its Heterogeneous Series type.

!!! warning
    This type's primary purpose is for iterating through the rows of a
    DataFrame with different column types. You should not attempt to
    directly create Series with mixed types.


Heterogeneous Series operations are a subset of those supported for
Series and the supported operations are listed below.

### Attributes


- [`pd.Series.index`][pdseriesindex]                                       
- [`pd.Series.values`][pdseriesvalues]                                     
- [`pd.Series.shape`][pdseriesshape]                                       
- [`pd.Series.ndim`][pdseriesndim]                                         
- [`pd.Series.size`][pdseriessize]                                         
- [`pd.Series.T`][pdseriest]                                               
- [`pd.Series.empty`][pdseriesempty]                                       
- [`pd.Series.name`][pdseriesname]                                         
