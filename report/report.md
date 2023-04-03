# Returns vs. 10-K Sentiment
By: Taylor Sheridan

__________________________________________________________________________________________________

## Summary

This report aims to answer the question of whether the positive or negative sentiment in a 10-K is associated with better/worse stock returns. In order to explore this question, I followed the steps listed below:

- Downloaded data on sp500 companies from wikipedia
- Download each firm's 2022 10-K file and put them in a zip folder
- Clean and store each 10-K's html text
- Create positive/negative sentiment dictionary, as well as contextual sentiment topic lists
- Use regex function to create sentiment scores for each firm's 10-K
- Use CIK and accession number to grab the filing date of each firm's 10-K
- Download 2022 stock returns
- Merge sentiment dataset with return dataset on symbol and filing date
- Download 2021 ccm accounting data
- Merge ccm dataset on Symbol to create dataset for analysis

After I collected my data and created a master dataset, I explored the relationships between a firm's stock returns on its 10-K filing date and the sentiment of its 10-K. From my findings, I discovered very weak relationships between stock returns and sentiment scores. This has mostly to do with incomplete data which I will explain within the data section of this report. However, I still managed to gain some insights into this relationship. 

## Data

The sample used for this analysis is firms within the sp500. Using the steps listed above, each firm's return on its 10-K filing date was added to the dataset, as well as sentiment variables using a regex function. Finally, this data was combined with ccm accounting data for additional firm statistics. 


The intended return variables for this assignment were to capture firm returns 2 days after the 10-K release, and returns between day 3-10 after the release. *Unfortunately*, I was unable to accomplish this, but below is my code to grab cumulative firm returns:

`analysis_df['cum_ret'] = analysis_df.assign(RET=1+clean_df['ret']).groupby('Symbol')['RET'].cumprod()`

This is only the code to get the cumulative returns for each firm, which would be the first step, but I was unable to figure out how to grab the returns for those two time periods around the 10-K filing date and store them in a new variable. I assume `.transform()` would have been useful.

The next step was to create sentiment variables for each firm's 10-K. I created 10 variables to score the file's positive or negative sentiment, as well as sentiment towards certain topics. Below is an example of how I created one of the sentiment variables, 'ML_Negative', which is sentiment using a list of negative words derived from machine learning:

```
with open('inputs/ML_negative_unigram.txt', 'r') as myfile:
    BHR_negative = [line.strip() for line in myfile]                # creates negative word list
    
    
BHR_negative_regex = '(' + '|'.join(BHR_negative) + ')'             # formats properly for regex function
regex1 = NEAR_regex([BHR_negative_regex])`                          # insert into regex function

for index, row in tqdm(firms_df1.iterrows()):                       # for loop for all firms
        
    doc_length = len(row['clean_html'].split())                     # stores length of file
        
    ML_negative_words = len(re.findall(regex1, row['clean_html']))  # finds all negative words from list within file
    BHR_negative_score = ML_negative_words / doc_length             # divide by length to get score
    firms_df1.loc[index, 'ML_Negative'] = BHR_negative_score        # store in variable
```

In addition to the positive/negative sentiment scores, I chose to explore how 3 topics were discussed within each 10-K to see if those individual topics affected stock price movement more. The three topics I chose were "covid", "inflation", and "innovation." I chose these topics because I thought they were relevant to business and the state of our economy. Covid-19 has been a hot topic of discussion in recent years because of its threat to people's lives, which both directly and indirectly affects business. I expected discussion on covid to have a negative impact on stock price. I also chose inflation because the rise of interest rates has greatly affected the economy and companies are monitoring them closely to predict its future impact. I expected discussion on inflation to decrease stock price, but not by much. Finally, I chose innovation because companies are always looking to make positive change and become a front-runner in their respective industries. I anticipate conversation around innovation to have a positive impact on stock performance. 

I provided a table of summary statistics of my final analysis sample below:


```python
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

analysis_df = pd.read_csv('output/analysis_sample.csv')
analysis_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CIK</th>
      <th>ML_Negative</th>
      <th>ML_Positive</th>
      <th>LM_Negative</th>
      <th>LM_Positive</th>
      <th>Covid_Negative</th>
      <th>Covid_Positive</th>
      <th>Inflation_Negative</th>
      <th>Inflation_Positive</th>
      <th>Innovation_Negative</th>
      <th>Innovation_Positive</th>
      <th>ret</th>
      <th>gvkey</th>
      <th>fyear</th>
      <th>lpermno</th>
      <th>lpermco</th>
      <th>sic</th>
      <th>sic3</th>
      <th>td</th>
      <th>long_debt_dum</th>
      <th>me</th>
      <th>l_a</th>
      <th>l_sale</th>
      <th>capx_a</th>
      <th>div_d</th>
      <th>age</th>
      <th>atr</th>
      <th>smalltaxlosscarry</th>
      <th>largetaxlosscarry</th>
      <th>gdpdef</th>
      <th>l_reala</th>
      <th>l_reallongdebt</th>
      <th>kz_index</th>
      <th>ww_index</th>
      <th>hp_index</th>
      <th>ww_unconstrain</th>
      <th>ww_constrained</th>
      <th>kz_unconstrain</th>
      <th>kz_constrained</th>
      <th>hp_unconstrain</th>
      <th>hp_constrained</th>
      <th>tnic3tsimm</th>
      <th>tnic3hhi</th>
      <th>prodmktfluid</th>
      <th>delaycon</th>
      <th>equitydelaycon</th>
      <th>debtdelaycon</th>
      <th>privdelaycon</th>
      <th>at_raw</th>
      <th>raw_Inv</th>
      <th>raw_Ch_Cash</th>
      <th>raw_Div</th>
      <th>raw_Ch_Debt</th>
      <th>raw_Ch_Eqty</th>
      <th>raw_Ch_WC</th>
      <th>raw_CF</th>
      <th>l_emp</th>
      <th>l_ppent</th>
      <th>l_laborratio</th>
      <th>Inv</th>
      <th>Ch_Cash</th>
      <th>Div</th>
      <th>Ch_Debt</th>
      <th>Ch_Eqty</th>
      <th>Ch_WC</th>
      <th>CF</th>
      <th>td_a</th>
      <th>td_mv</th>
      <th>mb</th>
      <th>prof_a</th>
      <th>ppe_a</th>
      <th>cash_a</th>
      <th>xrd_a</th>
      <th>dltt_a</th>
      <th>invopps_FG09</th>
      <th>sales_g</th>
      <th>dv_a</th>
      <th>short_debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.040000e+02</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>404.000000</td>
      <td>296.000000</td>
      <td>296.0</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>295.000000</td>
      <td>295.000000</td>
      <td>296.000000</td>
      <td>296.0</td>
      <td>2.960000e+02</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>225.000000</td>
      <td>225.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>275.000000</td>
      <td>295.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>295.000000</td>
      <td>296.000000</td>
      <td>275.000000</td>
      <td>296.0</td>
      <td>296.0</td>
      <td>261.000000</td>
      <td>261.000000</td>
      <td>259.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
      <td>277.000000</td>
      <td>295.000000</td>
      <td>296.000000</td>
      <td>296.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.911027e+05</td>
      <td>0.026199</td>
      <td>0.024079</td>
      <td>0.016107</td>
      <td>0.005046</td>
      <td>0.000431</td>
      <td>0.000182</td>
      <td>0.000269</td>
      <td>0.000142</td>
      <td>0.000911</td>
      <td>0.000686</td>
      <td>0.167894</td>
      <td>43783.293919</td>
      <td>2021.0</td>
      <td>53033.912162</td>
      <td>26228.320946</td>
      <td>4277.701695</td>
      <td>427.545763</td>
      <td>14267.287520</td>
      <td>1.0</td>
      <td>9.182268e+04</td>
      <td>9.980973</td>
      <td>9.506013</td>
      <td>0.029643</td>
      <td>0.746622</td>
      <td>1.986486</td>
      <td>0.212397</td>
      <td>0.711111</td>
      <td>0.226667</td>
      <td>121.574561</td>
      <td>5.180524</td>
      <td>3.904354</td>
      <td>-6.637216</td>
      <td>-0.352671</td>
      <td>-2.690920</td>
      <td>0.787162</td>
      <td>0.088136</td>
      <td>0.351351</td>
      <td>0.214545</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.767452</td>
      <td>0.325586</td>
      <td>3.203900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41881.764321</td>
      <td>0.065606</td>
      <td>-0.009950</td>
      <td>0.023712</td>
      <td>0.008183</td>
      <td>-0.046383</td>
      <td>0.017259</td>
      <td>0.117568</td>
      <td>3.343312</td>
      <td>8.107116</td>
      <td>4.823714</td>
      <td>0.065606</td>
      <td>-0.009950</td>
      <td>0.023712</td>
      <td>0.007255</td>
      <td>-0.044298</td>
      <td>0.015921</td>
      <td>0.117568</td>
      <td>0.349147</td>
      <td>0.181725</td>
      <td>3.480484</td>
      <td>0.153559</td>
      <td>0.231585</td>
      <td>0.133054</td>
      <td>0.028327</td>
      <td>0.321326</td>
      <td>3.128103</td>
      <td>0.291556</td>
      <td>0.023712</td>
      <td>0.089924</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.569934e+05</td>
      <td>0.003227</td>
      <td>0.003492</td>
      <td>0.003711</td>
      <td>0.001354</td>
      <td>0.000259</td>
      <td>0.000124</td>
      <td>0.000162</td>
      <td>0.000094</td>
      <td>0.000302</td>
      <td>0.000221</td>
      <td>3.584079</td>
      <td>59711.952251</td>
      <td>0.0</td>
      <td>30077.168794</td>
      <td>16824.282114</td>
      <td>1945.905139</td>
      <td>194.622792</td>
      <td>23043.292915</td>
      <td>0.0</td>
      <td>2.390276e+05</td>
      <td>1.108001</td>
      <td>1.194087</td>
      <td>0.024414</td>
      <td>0.435682</td>
      <td>0.141971</td>
      <td>0.182540</td>
      <td>0.454257</td>
      <td>0.419609</td>
      <td>1.533136</td>
      <td>1.107450</td>
      <td>1.292392</td>
      <td>8.495044</td>
      <td>0.336028</td>
      <td>0.314564</td>
      <td>0.410007</td>
      <td>0.283974</td>
      <td>0.478201</td>
      <td>0.411255</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.033720</td>
      <td>0.272094</td>
      <td>1.737992</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>64008.097754</td>
      <td>0.089279</td>
      <td>0.055106</td>
      <td>0.025755</td>
      <td>0.084213</td>
      <td>0.067102</td>
      <td>0.062267</td>
      <td>0.087549</td>
      <td>1.096886</td>
      <td>1.438016</td>
      <td>1.346984</td>
      <td>0.089279</td>
      <td>0.055106</td>
      <td>0.025755</td>
      <td>0.077423</td>
      <td>0.059943</td>
      <td>0.048565</td>
      <td>0.087549</td>
      <td>0.189848</td>
      <td>0.143796</td>
      <td>2.747725</td>
      <td>0.083325</td>
      <td>0.203483</td>
      <td>0.122334</td>
      <td>0.042566</td>
      <td>0.183141</td>
      <td>2.784793</td>
      <td>0.868479</td>
      <td>0.025755</td>
      <td>0.089011</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.488000e+03</td>
      <td>0.008953</td>
      <td>0.003546</td>
      <td>0.006875</td>
      <td>0.001773</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-24.277852</td>
      <td>1045.000000</td>
      <td>2021.0</td>
      <td>10104.000000</td>
      <td>7.000000</td>
      <td>100.000000</td>
      <td>10.000000</td>
      <td>60.067000</td>
      <td>1.0</td>
      <td>6.559703e+03</td>
      <td>7.592752</td>
      <td>6.836294</td>
      <td>0.001387</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>117.922000</td>
      <td>2.791126</td>
      <td>0.329143</td>
      <td>-50.967920</td>
      <td>-0.647860</td>
      <td>-3.230299</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.036273</td>
      <td>0.457329</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1983.764000</td>
      <td>-0.239829</td>
      <td>-0.324262</td>
      <td>0.000000</td>
      <td>-0.206671</td>
      <td>-0.357998</td>
      <td>-0.150786</td>
      <td>-0.613519</td>
      <td>0.625938</td>
      <td>4.580744</td>
      <td>0.519750</td>
      <td>-0.239829</td>
      <td>-0.324262</td>
      <td>0.000000</td>
      <td>-0.206671</td>
      <td>-0.223117</td>
      <td>-0.150786</td>
      <td>-0.613519</td>
      <td>0.006418</td>
      <td>0.000676</td>
      <td>0.878375</td>
      <td>-0.077358</td>
      <td>0.013654</td>
      <td>0.004218</td>
      <td>0.000000</td>
      <td>0.004913</td>
      <td>0.481436</td>
      <td>-0.658981</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.767775e+04</td>
      <td>0.024186</td>
      <td>0.021882</td>
      <td>0.013652</td>
      <td>0.004092</td>
      <td>0.000242</td>
      <td>0.000094</td>
      <td>0.000153</td>
      <td>0.000073</td>
      <td>0.000702</td>
      <td>0.000541</td>
      <td>-1.618724</td>
      <td>6420.000000</td>
      <td>2021.0</td>
      <td>19474.750000</td>
      <td>13972.750000</td>
      <td>2843.000000</td>
      <td>284.000000</td>
      <td>3256.413000</td>
      <td>1.0</td>
      <td>1.900649e+04</td>
      <td>9.250529</td>
      <td>8.636038</td>
      <td>0.012694</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.125560</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>121.708000</td>
      <td>4.461192</td>
      <td>3.201480</td>
      <td>-10.787620</td>
      <td>-0.495996</td>
      <td>-2.929592</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.101900</td>
      <td>0.130472</td>
      <td>2.006005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10410.080000</td>
      <td>0.021447</td>
      <td>-0.031603</td>
      <td>0.000000</td>
      <td>-0.029409</td>
      <td>-0.068264</td>
      <td>-0.003505</td>
      <td>0.069272</td>
      <td>2.555745</td>
      <td>7.054157</td>
      <td>4.001654</td>
      <td>0.021447</td>
      <td>-0.031603</td>
      <td>0.000000</td>
      <td>-0.029409</td>
      <td>-0.068264</td>
      <td>-0.003505</td>
      <td>0.069272</td>
      <td>0.227240</td>
      <td>0.072934</td>
      <td>1.647529</td>
      <td>0.099179</td>
      <td>0.089342</td>
      <td>0.041530</td>
      <td>0.000000</td>
      <td>0.200364</td>
      <td>1.350073</td>
      <td>0.085556</td>
      <td>0.000000</td>
      <td>0.026400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.853060e+05</td>
      <td>0.026048</td>
      <td>0.024150</td>
      <td>0.015948</td>
      <td>0.004957</td>
      <td>0.000380</td>
      <td>0.000152</td>
      <td>0.000241</td>
      <td>0.000121</td>
      <td>0.000866</td>
      <td>0.000662</td>
      <td>-0.096874</td>
      <td>13710.500000</td>
      <td>2021.0</td>
      <td>57737.000000</td>
      <td>21169.000000</td>
      <td>3728.000000</td>
      <td>372.000000</td>
      <td>6772.000000</td>
      <td>1.0</td>
      <td>3.463996e+04</td>
      <td>9.917810</td>
      <td>9.440016</td>
      <td>0.022716</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.193019</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>121.708000</td>
      <td>5.116185</td>
      <td>3.938389</td>
      <td>-3.761360</td>
      <td>-0.457291</td>
      <td>-2.725088</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.363700</td>
      <td>0.236235</td>
      <td>2.883351</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20288.500000</td>
      <td>0.044525</td>
      <td>-0.003761</td>
      <td>0.017509</td>
      <td>-0.001578</td>
      <td>-0.023804</td>
      <td>0.009987</td>
      <td>0.106095</td>
      <td>3.246491</td>
      <td>8.046027</td>
      <td>4.530496</td>
      <td>0.044525</td>
      <td>-0.003761</td>
      <td>0.017509</td>
      <td>-0.001578</td>
      <td>-0.023804</td>
      <td>0.009987</td>
      <td>0.106095</td>
      <td>0.321847</td>
      <td>0.153236</td>
      <td>2.535912</td>
      <td>0.137471</td>
      <td>0.158500</td>
      <td>0.097425</td>
      <td>0.009919</td>
      <td>0.299592</td>
      <td>2.170944</td>
      <td>0.158671</td>
      <td>0.017509</td>
      <td>0.061669</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.136875e+06</td>
      <td>0.028200</td>
      <td>0.026164</td>
      <td>0.018106</td>
      <td>0.005801</td>
      <td>0.000572</td>
      <td>0.000252</td>
      <td>0.000347</td>
      <td>0.000188</td>
      <td>0.001068</td>
      <td>0.000806</td>
      <td>1.798563</td>
      <td>61435.750000</td>
      <td>2021.0</td>
      <td>82546.750000</td>
      <td>40395.750000</td>
      <td>5331.000000</td>
      <td>533.000000</td>
      <td>14422.000000</td>
      <td>1.0</td>
      <td>7.192174e+04</td>
      <td>10.702927</td>
      <td>10.170132</td>
      <td>0.038324</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.235804</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>121.708000</td>
      <td>5.891841</td>
      <td>4.667277</td>
      <td>-0.574875</td>
      <td>-0.378925</td>
      <td>-2.507755</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.140700</td>
      <td>0.432629</td>
      <td>4.150276</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44485.916000</td>
      <td>0.084956</td>
      <td>0.013003</td>
      <td>0.037061</td>
      <td>0.020383</td>
      <td>-0.000694</td>
      <td>0.026170</td>
      <td>0.161380</td>
      <td>4.191917</td>
      <td>9.094283</td>
      <td>5.453990</td>
      <td>0.084956</td>
      <td>0.013003</td>
      <td>0.037061</td>
      <td>0.020383</td>
      <td>-0.000694</td>
      <td>0.026170</td>
      <td>0.161380</td>
      <td>0.445578</td>
      <td>0.245099</td>
      <td>4.285134</td>
      <td>0.197564</td>
      <td>0.309560</td>
      <td>0.179042</td>
      <td>0.041274</td>
      <td>0.405472</td>
      <td>3.701127</td>
      <td>0.290145</td>
      <td>0.037061</td>
      <td>0.123019</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.868275e+06</td>
      <td>0.038030</td>
      <td>0.037982</td>
      <td>0.030185</td>
      <td>0.010899</td>
      <td>0.001429</td>
      <td>0.000793</td>
      <td>0.001075</td>
      <td>0.000603</td>
      <td>0.002026</td>
      <td>0.001557</td>
      <td>16.214105</td>
      <td>316056.000000</td>
      <td>2021.0</td>
      <td>93132.000000</td>
      <td>58235.000000</td>
      <td>8742.000000</td>
      <td>874.000000</td>
      <td>177930.000000</td>
      <td>1.0</td>
      <td>2.324390e+06</td>
      <td>12.949316</td>
      <td>13.253324</td>
      <td>0.157599</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>126.907000</td>
      <td>8.147692</td>
      <td>7.222624</td>
      <td>3.794306</td>
      <td>1.055742</td>
      <td>-1.802074</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>80.969100</td>
      <td>1.000000</td>
      <td>11.815062</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>420549.000000</td>
      <td>0.586530</td>
      <td>0.171227</td>
      <td>0.163095</td>
      <td>0.734575</td>
      <td>0.133418</td>
      <td>0.780077</td>
      <td>0.552110</td>
      <td>5.247024</td>
      <td>10.627358</td>
      <td>9.737904</td>
      <td>0.586530</td>
      <td>0.171227</td>
      <td>0.163095</td>
      <td>0.460031</td>
      <td>0.133418</td>
      <td>0.390622</td>
      <td>0.552110</td>
      <td>1.161385</td>
      <td>0.798769</td>
      <td>14.733148</td>
      <td>0.405925</td>
      <td>0.888302</td>
      <td>0.607837</td>
      <td>0.258595</td>
      <td>1.019505</td>
      <td>14.066011</td>
      <td>14.183099</td>
      <td>0.163095</td>
      <td>0.530059</td>
    </tr>
  </tbody>
</table>
</div>



After taking a first look at my data and some analysis, I did not notice anything unusual; however, this is because I understand my data is incomplete and this will be explained below. One thing I would point out is how low my sentiment scores are. This likely means that either my topics did not receive a lot of hits, or I made an error.

### Data Warning !!!

When assembling my data, I ran into a few issues that greatly impacted the results of my analysis. The most significant error is that I was not able to get return variables for each firm 2 days after the 10-K filing, and 3-10 days after. I only used the firm's returns on that day of trading. This would have been sufficient to identify *some relationship* between stock returns and 10-K sentiment, but 10-K's are released at different times of day, meaning the return on that day can be caused by many different factors. For example, a firm may not release their 10-K until 4pm, but trading is already finished, so the returns in my data are not related to the 10-K sentiment, other than insider trading, rumors, etc...

Another problem with my data is that it only represents 405 firms in the sp500. When downloading each firm's 10-K html file, I only grabbed 405, and so I dropped the rest. I still belive 80% of the population would be enough to make some kind of conclusion about the relationship, but it wouldn't be as complete or accurate, especially if some of the top firms were missing.

## Results

In order to analyze the relationship between a firm's stock return on its 10-K filing date and the sentiment of its 10-K, I created scatterplots of return vs. sentiment score. I also included correlation coefficients on each graph to describe the strength of the relationship. The code I wrote and its output is provided below:


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

analysis_df = pd.read_csv('output/analysis_sample.csv')
a1 = analysis_df[['Symbol','ret','ML_Negative','ML_Positive','LM_Negative','LM_Positive','Covid_Negative','Covid_Positive','Inflation_Negative','Inflation_Positive','Innovation_Negative','Innovation_Positive']]

plt.subplots(figsize = ( 10 , 10 )) 
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=1.0)


plt.subplot(5, 2, 1) # row 1, col 2 index 1
plt.scatter(a1['ML_Negative'], a1['ret'], s = 10, alpha = 0.2, color = 'red')
plt.title("ML_Negative", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylabel('return')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['ML_Negative']), np.poly1d(np.polyfit(a1['ML_Negative'], a1['ret'], 1))(np.unique(a1['ML_Negative'])))
r = np.round(np.corrcoef(a1['ML_Negative'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')

plt.subplot(5, 2, 2) # index 2
plt.scatter(a1['ML_Positive'], a1['ret'], s = 10, alpha = 0.2, color = 'green')
plt.title("ML_Positive", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['ML_Positive']), np.poly1d(np.polyfit(a1['ML_Positive'], a1['ret'], 1))(np.unique(a1['ML_Positive'])))
r = np.round(np.corrcoef(a1['ML_Positive'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')

plt.subplot(5, 2, 3) # row 1, col 2 index 1
plt.scatter(a1['LM_Negative'], a1['ret'], s = 10, alpha = 0.2, color = 'red')
plt.title("LM_Negative", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylabel('return')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['LM_Negative']), np.poly1d(np.polyfit(a1['LM_Negative'], a1['ret'], 1))(np.unique(a1['LM_Negative'])))
r = np.round(np.corrcoef(a1['LM_Negative'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')

plt.subplot(5, 2, 4) # index 2
plt.scatter(a1['LM_Positive'], a1['ret'], s = 10, alpha = 0.2, color = 'green')
plt.title("LM_Positive", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['LM_Positive']), np.poly1d(np.polyfit(a1['LM_Positive'], a1['ret'], 1))(np.unique(a1['LM_Positive'])))
r = np.round(np.corrcoef(a1['LM_Positive'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')

plt.subplot(5, 2, 5) # row 1, col 2 index 1
plt.scatter(a1['Covid_Negative'], a1['ret'], s = 10, alpha = 0.2, color = 'red')
plt.title("Covid_Negative", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylabel('return')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['Covid_Negative']), np.poly1d(np.polyfit(a1['Covid_Negative'], a1['ret'], 1))(np.unique(a1['Covid_Negative'])))
r = np.round(np.corrcoef(a1['Covid_Negative'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')

plt.subplot(5, 2, 6) # index 2
plt.scatter(a1['Covid_Positive'], a1['ret'], s = 10, alpha = 0.2, color = 'green')
plt.title("Covid_Positive", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['Covid_Positive']), np.poly1d(np.polyfit(a1['Covid_Positive'], a1['ret'], 1))(np.unique(a1['Covid_Positive'])))
r = np.round(np.corrcoef(a1['Covid_Positive'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')

plt.subplot(5, 2, 7) # row 1, col 2 index 1
plt.scatter(a1['Inflation_Negative'], a1['ret'], s = 10, alpha = 0.2, color = 'red')
plt.title("Inflation_Negative", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylabel('return')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['Inflation_Negative']), np.poly1d(np.polyfit(a1['Inflation_Negative'], a1['ret'], 1))(np.unique(a1['Inflation_Negative'])))
r = np.round(np.corrcoef(a1['Inflation_Negative'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')

plt.subplot(5, 2, 8) # index 2
plt.scatter(a1['Inflation_Positive'], a1['ret'], s = 10, alpha = 0.2, color = 'green')
plt.title("Inflation_Positive", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['Inflation_Positive']), np.poly1d(np.polyfit(a1['Inflation_Positive'], a1['ret'], 1))(np.unique(a1['Inflation_Positive'])))
r = np.round(np.corrcoef(a1['Inflation_Positive'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')

plt.subplot(5, 2, 9) # row 1, col 2 index 1
plt.scatter(a1['Innovation_Negative'], a1['ret'], s = 10, alpha = 0.2, color = 'red')
plt.title("Innovation_Negative", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylabel('return')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['Innovation_Negative']), np.poly1d(np.polyfit(a1['Innovation_Negative'], a1['ret'], 1))(np.unique(a1['Innovation_Negative'])))
r = np.round(np.corrcoef(a1['Innovation_Negative'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')

plt.subplot(5, 2, 10) # index 2
plt.scatter(a1['Innovation_Positive'], a1['ret'], s = 10, alpha = 0.2, color = 'green')
plt.title("Innovation_Positive", fontsize = 16)
plt.xlabel('sentiment score')
plt.ylim(-10, 10)
plt.plot(np.unique(a1['Innovation_Positive']), np.poly1d(np.polyfit(a1['Innovation_Positive'], a1['ret'], 1))(np.unique(a1['Innovation_Positive'])))
r = np.round(np.corrcoef(a1['Innovation_Positive'], a1['ret'])[0,1], 3)
plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.7), xycoords='axes fraction')
```

    /var/folders/xf/_j35z7hx68l2fyn2q50zvmlr0000gn/T/ipykernel_53932/2042368599.py:18: MatplotlibDeprecationWarning: Auto-removal of overlapping axes is deprecated since 3.6 and will be removed two minor releases later; explicitly call ax.remove() as needed.
      plt.subplot(5, 2, 1) # row 1, col 2 index 1





    Text(0.8, 0.7, 'r = -0.02')




    
![png](output_14_2.png)
    


### Discusssion Topics



### 1)

Most notably, the both the ML positive and negative lists received more regex hits than the LM list. I believe the ML list contains more words, which could be a factor; however, I also believe this makes sense because it seems reasonable that a computer gathering data would be more accurate than a list a human created. 

The ML sentiment had a positive corrleation with stock price for both the positve and negative list, although very weak. Oppositely, the LM sentiment had a weaker, relationship with r = 0 for LM negative and r =- -0.06 for LM positve. 

### 2)

My results conflict with those of Table 3 within the Garcia, Hu, and Rohrer paper (ML_JFE.pdf, in the repo). Their chart represents much stronger relationships between returns and 10-K sentiment. Again, this is due to my failure to obtain the appropriate return variables.

### 3)

None of my conceptual sentiment measures indicated a strong relationship with stock returns, but I did notice a patten within the nature of the word despite *how* they were talked about. More specifically, the words "covid" and "inflation" had an overall negative relationship with stock returns, indicating that mentioning these words at all drove prices down, despite being discussed in a positive or negative manner. There isnt enough to make a conclusion on this, but I felt it was worth pointing out and that it makes reaonable sense.

### 4)

From my scatterplot, there is little difference between the relationship of stock returns and ML positive and negative sentiment. This likely suggests that the positive words in the ML list occur just as much as the negative words.
