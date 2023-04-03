---
layout: wide_default
---

# Read this part

This webpage is created from a file we made in a class. To convert it to a webpage, 
1. In JupyterLab, click File, then "Export Notebook as", then markdown.
2. Add that file into this repo. 
3. Edit that file. At the top of the file, add 3 lines with exactly this and save/commit/push it.
    ```
    ---
    layout: wide_default
    ---    
    
    ```
    These lines make the report part of the webpage wider, which is usually a good idea for reports. 
    
    _Note 1: You can do this on any page in the website._
    
    _Note 2: The site won't look great on Mobile._
4. Naturally, you'll want to add a link to the new page you just made in your portfolio section on the main page.

- [**Read the chapter on the website!**](https://ledatascifi.github.io/ledatascifi-2022/content/05/02_reg.html) It contains a lot of extra information we won't cover in class extensively.
- After reading that, I recommend [this webpage as a complimentary place to get additional intuition.](https://aeturrell.github.io/coding-for-economists/econmt-regression.html)

---

## Today

[Finish picking teams and declare initial project interests in the project sheet](https://docs.google.com/spreadsheets/d/1A6oQfhTBHHEb_EWSgfQv2KgsBoZm4V_BQWuvTjxnrdo/edit#gid=1508330834)


# Today is mostly about INTERPRETING COEFFICIENTS (5.2.4 in the book)

1. 25 min reading groups: Talk/read through two regression pages (5.2.3 and 5.2.4) 
    - Assemble your own notes. Perhaps in the "Module 4 notes" file, but you can do this in any file you want.
    - After class, each group will email their notes to Julio/me for participation. (Effort grading.)
1. 10 min: class builds joint "big takeaways and nuanced observations" 
1. 5 min: Interpret models 1-2 as class as practice. 
1. 20 min reading groups: Work through remaining problems below.
1. 10 min: wrap up  

---

## Our notes (920am class)

- R2 is the % variation in y captured by the model
- You can never DECREASE r2 by adding variables to the model... but this is a problem, leads to overfitting and even adding "random noise" variables
- Instead - focus on Adj R2: it penalizes adding adtl variables

- John/Theo R2: % of variation in y that our model explains
    - "higher is better"
    - when you are vars to a model, R2 can't decrease... tendency to overfit
    - so use Adj R2: R2 with a penalty for additional variables
    
- Ian: by transforming variables, we change the interpretation of the relationship
    - logs, polynomails, etc
    
- Colin: Categorical variables: a regression will give you a coefficient for each level EXCEPT ONE
    - The coefficients on the levels in the reg are "relative" to the omitted group
    
- interpration is mechanical 
    - A (1 unit) increase in X is assoc with a ( ) change in Y, holding all else constant



```python
import pandas as pd
from statsmodels.formula.api import ols as sm_ols
import numpy as np
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col # nicer tables

```


```python
url = 'https://github.com/LeDataSciFi/ledatascifi-2022/blob/main/data/Fannie_Mae_Plus_Data.gzip?raw=true'
fannie_mae = pd.read_csv(url,compression='gzip') 
```

## Clean the data and create variables you want


```python
fannie_mae = (fannie_mae
                  # create variables
                  .assign(l_credscore = np.log(fannie_mae['Borrower_Credit_Score_at_Origination']),
                          l_LTV = np.log(fannie_mae['Original_LTV_(OLTV)']),
                          l_int = np.log(fannie_mae['Original_Interest_Rate']),
                          Origination_Date = lambda x: pd.to_datetime(x['Origination_Date']),
                          Origination_Year = lambda x: x['Origination_Date'].dt.year,
                          const = 1
                         )
                  .rename(columns={'Original_Interest_Rate':'int'}) # shorter name will help the table formatting
             )

# create a categorical credit bin var with "pd.cut()"
fannie_mae['creditbins']= pd.cut(fannie_mae['Co-borrower_credit_score_at_origination'],
                                 [0,579,669,739,799,850],
                                 labels=['Very Poor','Fair','Good','Very Good','Exceptional'])

```


```python
fannie_mae['Borrower_Credit_Score_at_Origination'].describe()
```




    count    134481.000000
    mean        742.428797
    std          53.428076
    min         361.000000
    25%         707.000000
    50%         755.000000
    75%         786.000000
    max         850.000000
    Name: Borrower_Credit_Score_at_Origination, dtype: float64



## Statsmodels

As before, the psuedocode:
```python
model = sm_ols(<formula>, data=<dataframe>)
result=model.fit()

# you use result to print summary, get predicted values (.predict) or residuals (.resid)
```

Now, let's save each regression's result with a different name, and below this, output them all in one nice table:


```python
# one var: 'y ~ x' means fit y = a + b*X

reg1 = sm_ols('int ~  Borrower_Credit_Score_at_Origination ', data=fannie_mae).fit()

reg1b= sm_ols('int ~  l_credscore  ',  data=fannie_mae).fit()

reg1c= sm_ols('l_int ~  Borrower_Credit_Score_at_Origination  ',  data=fannie_mae).fit()

reg1d= sm_ols('l_int ~  l_credscore  ',  data=fannie_mae).fit()

# multiple variables: just add them to the formula
# 'y ~ x1 + x2' means fit y = a + b*x1 + c*x2
reg2 = sm_ols('int ~  l_credscore + l_LTV ',  data=fannie_mae).fit()

# interaction terms: Just use *
# Note: always include each variable separately too! (not just x1*x2, but x1+x2+x1*x2)
reg3 = sm_ols('int ~  l_credscore + l_LTV + l_credscore*l_LTV',  data=fannie_mae).fit()
      
# categorical dummies: C() 
reg4 = sm_ols('int ~  C(creditbins)  ',  data=fannie_mae).fit()

reg5 = sm_ols('int ~  C(creditbins)  -1', data=fannie_mae).fit()

```

Ok, time to output them:


```python
# now I'll format an output table
# I'd like to include extra info in the table (not just coefficients)
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'Adj R-squared' : lambda x: f"{x.rsquared_adj:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

# q4b1 and q4b2 name the dummies differently in the table, so this is a silly fix
reg4.model.exog_names[1:] = reg5.model.exog_names[1:]  

# This summary col function combines a bunch of regressions into one nice table
print('='*108)
print('                  y = interest rate if not specified, log(interest rate else)')
print(summary_col(results=[reg1,reg1b,reg1c,reg1d,reg2,reg3,reg4,reg5], # list the result obj here
                  float_format='%0.2f',
                  stars = True, # stars are easy way to see if anything is statistically significant
                  model_names=['1','2',' 3 (log)','4 (log)','5','6','7','8'], # these are bad names, lol. Usually, just use the y variable name
                  info_dict=info_dict,
                  regressor_order=[ 'Intercept','Borrower_Credit_Score_at_Origination','l_credscore','l_LTV','l_credscore:l_LTV',
                                  'C(creditbins)[Very Poor]','C(creditbins)[Fair]','C(creditbins)[Good]','C(creditbins)[Vrey Good]','C(creditbins)[Exceptional]']
                  )
     )
```

    ============================================================================================================
                      y = interest rate if not specified, log(interest rate else)
    
    ============================================================================================================
                                            1        2      3 (log) 4 (log)     5         6        7        8   
    ------------------------------------------------------------------------------------------------------------
    Intercept                            11.58*** 45.37*** 2.87***  9.50***  44.13*** -16.81*** 6.65***         
                                         (0.05)   (0.29)   (0.01)   (0.06)   (0.30)   (4.11)    (0.08)          
    Borrower_Credit_Score_at_Origination -0.01***          -0.00***                                             
                                         (0.00)            (0.00)                                               
    l_credscore                                   -6.07***          -1.19*** -5.99*** 3.22***                   
                                                  (0.04)            (0.01)   (0.04)   (0.62)                    
    l_LTV                                                                    0.15***  14.61***                  
                                                                             (0.01)   (0.97)                    
    l_credscore:l_LTV                                                                 -2.18***                  
                                                                                      (0.15)                    
    C(creditbins)[Very Poor]                                                                             6.65***
                                                                                                         (0.08) 
    C(creditbins)[Fair]                                                                         -0.63*** 6.02***
                                                                                                (0.08)   (0.02) 
    C(creditbins)[Good]                                                                         -1.17*** 5.48***
                                                                                                (0.08)   (0.01) 
    C(creditbins)[Exceptional]                                                                  -2.25*** 4.40***
                                                                                                (0.08)   (0.01) 
    C(creditbins)[Very Good]                                                                    -1.65*** 5.00***
                                                                                                (0.08)   (0.01) 
    R-squared                            0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    R-squared Adj.                       0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    R-squared                            0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    Adj R-squared                        0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    No. observations                     134481   134481   134481   134481   134481   134481    67366    67366  
    ============================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01
    


```python
fannie_mae['int'].describe()
```




    count    135038.000000
    mean          5.238376
    std           1.289895
    min           2.250000
    25%           4.250000
    50%           5.250000
    75%           6.125000
    max          11.000000
    Name: int, dtype: float64



# Today. Work in groups. Refer to the lectures. 

You might need to print out a few individual regressions with more decimals.

1. Interpret coefs in model 1-4
    - Model 1: The predicted interest rate for borrowers with a credit score of 0 is 11.5%.
    - Model 1: A 1 unit inc in credit score is assoc with a 1 b.p. decrease in interest rates, holding all other X constant.
        - If credit score goes from 700 to 707: rate falls 7 b.p.
        - If credit score goes from 600 to 606: rate falls 6 b.p.
    - Model 2: A 1% inc in credit score is assoc with a 6.07 b.p. decrease in interest rates, holding all other X constant.
        - If credit score goes from 700 to 707: rate falls 6.07 b.p.
        - If credit score goes from 600 to 606: rate falls 6.07 b.p.
    - Model 3: A 1 unit inc in credit score is assoc with a 0.17% (percent! not p.p.) decrease in interest rates, holding all other X constant.
        - If credit score goes from 700 to 707: rate falls by 6 b.p.   (assuming y starts at avg y of 5.23)
    - Model 4: A 1% inc in credit score is assoc with a 1.19% decrease in interest rates, holding all other X constant.
        - If credit score goes from 700 to 707: rate falls by 6 b.p.   (assuming y starts at avg y of 5.23)
1. Interpret coefs in model 5
    - Model 5: A 1% inc in credit score is assoc with a 5.99 b.p. decrease in interest rates, holding  constant  log_LTV.

1. Interpret coefs in model 6 (and visually?)
    - int = a + c*l_credscore + d*l_LTV + e*l_credscore*l_LTV
    - deriv of int w.r.t to credit score:       3.22 - 2.18 * log(LTV)
    - For loans with an average log(LTV) of 4.2= 3.22-2.18*4.2 = -5.936
    - A 1 % increase in credit score is assoc with interest rates that are 5.936 bp lower, ceterus paribus

1. Interpret coefs in model 7 (and visually? + comp to table)
1. Interpret coefs in model 8 (and visually? + comp to table)
1. Add l_LTV  to Model 8 and interpret (and visually?)




