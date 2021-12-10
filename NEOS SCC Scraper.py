
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:04:29 2021

@author: Tommaso
"""


#This FIle extracts AMPL Optimization Results from NEOS Knitro Browser Version and calculates The Social Costs of Carbon.



import requests 
import re
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
s = requests.Session()

def crawlURL(url_path):
    attempts=0
    pageContent=''
    global s
    while attempts<5:
        try:
            pageContent = s.get(url_path).text
            attempts = 10 
        except:
            attempts+=1;    
            print('crawl error URL')
            #ErrorFile.write(str(url_path)+"\n")
    return pageContent

Time_horizon=2500

# #BLOSS MODEL 1
    
rel=crawlURL('https://neos-server.org/neos/jobs/11050000/11054098.html') #1

soup=BeautifulSoup(rel, 'html.parser')
soupstring=str(soup)
full_text=soup.find('div',{'class':'body-container'}).text

acc=re.findall(r'constr_accounting \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=acc.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))


r=pd.DataFrame({'constr. accounting':var,'t':ind})
r=r.set_index('t')
r=r.sort_index()


em=re.findall(r'constr_emissions \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=em.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))


r1=pd.DataFrame({'constr. emissions':var,'t':ind})
r1=r1.set_index('t')
r1=r1.sort_index()


r=r.join(r1['constr. emissions'])
scc_1=-1000*r['constr. emissions']/r['constr. accounting']
r['scc']=scc_1



TT = np.linspace(2000, Time_horizon,100 , dtype = np.int32)
plt.style.use('seaborn')

import seaborn as sns

plt.title('Social Costs of Carbon')

sns.set_palette('Set1')
plt.title('Social Costs of Carbon')
plt.plot(TT[4:21], scc_1[4:21], label='\u03C0 = 1', color='#1A76A7')


print('bloss_1')
print(scc_1[4])
print(scc_1[20])


#BLOSS MODEL -1

    
rel=crawlURL('https://neos-server.org/neos/jobs/11050000/11054097.html') #https://neos-server.org/neos/jobs/10430000/10432898.html


soup=BeautifulSoup(rel, 'html.parser')
soupstring=str(soup)
full_text=soup.find('div',{'class':'body-container'}).text


acc=re.findall(r'constr_accounting \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=acc.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))


r=pd.DataFrame({'constr. accounting':var,'t':ind})
r=r.set_index('t')
r=r.sort_index()


em=re.findall(r'constr_emissions \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=em.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))


r1=pd.DataFrame({'constr. emissions':var,'t':ind})
r1=r1.set_index('t')
r1=r1.sort_index()


r=r.join(r1['constr. emissions'])
scc_m1=-1000*r['constr. emissions']/r['constr. accounting']
r['scc']=scc_m1
print('bloss_m1')
print(scc_m1[4])
print(scc_m1[20])

TT = np.linspace(2000, Time_horizon,100 , dtype = np.int32)


plt.title('Social Costs of Carbon')

plt.plot(TT[4:21], scc_m1[4: 21], label='\u03C0 = -2.5', color='red')




#NORDHAUS

rel=crawlURL('https://neos-server.org/neos/jobs/11050000/11054100.html')


soup=BeautifulSoup(rel, 'html.parser')
soupstring=str(soup)
full_text=soup.find('div',{'class':'body-container'}).text




acc=re.findall(r'constr_accounting \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=acc.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))
    

r=pd.DataFrame({'constr. accounting':var,'t':ind})
r=r.set_index('t')
r=r.sort_index()

em=re.findall(r'constr_emissions \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=em.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))


r1=pd.DataFrame({'constr. emissions':var,'t':ind})
r1=r1.set_index('t')
r1=r1.sort_index()


r=r.join(r1['constr. emissions'])
scc=-1000*r['constr. emissions']/r['constr. accounting']


print('Nord')
print(scc[4])
print(scc[20])

r['scc']=scc


TT = np.linspace(2000, Time_horizon,100 , dtype = np.int32)


plt.title('Social Costs of Carbon', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(TT[4:21], scc[4: 21], label='Nordhaus', color='green')

'''
#STERN MODEL 

    
rel=crawlURL('https://neos-server.org/neos/jobs/10780000/10780408.html')


soup=BeautifulSoup(rel, 'html.parser')
soupstring=str(soup)
full_text=soup.find('div',{'class':'body-container'}).text




acc=re.findall(r'constr_accounting \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=acc.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))


r=pd.DataFrame({'constr. accounting':var,'t':ind})
r=r.set_index('t')
r=r.sort_index()


em=re.findall(r'constr_emissions \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=em.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))


r1=pd.DataFrame({'constr. emissions':var,'t':ind})
r1=r1.set_index('t')
r1=r1.sort_index()


r=r.join(r1['constr. emissions'])
scc=-1000*r['constr. emissions']/r['constr. accounting']
r['scc']=scc


TT = np.linspace(2000, Time_horizon,100 , dtype = np.int32)


plt.title('Social Costs of Carbon', fontsize=18)
plt.plot(TT[4:21], scc[4: 21], label='Stern', color ='purple')
plt.xlabel('Years', fontsize=15)
plt.ylabel('US$/tCO2',fontsize=15)
#plt.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=4, fontsize=14)


print('stern')
print(scc[4])
print(scc[20])

'''
#REL PRICES


rel=crawlURL('https://neos-server.org/neos/jobs/11050000/11054095.html')


soup=BeautifulSoup(rel, 'html.parser')
soupstring=str(soup)
full_text=soup.find('div',{'class':'body-container'}).text




acc=re.findall(r'constr_accounting \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=acc.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))


r=pd.DataFrame({'constr. accounting':var,'t':ind})
r=r.set_index('t')
r=r.sort_index()


em=re.findall(r'constr_emissions \[\*] \:\=(.*?);',soupstring, re.DOTALL)[0].replace('\n',' ')
p=em.replace('\n',' ').split(' ');p

ind=[]
var=[]
l=[]

for i in range(len(p)):
    if p[i]!='':
        l.append(p[i])
for i in range(0,len(l),2):
    ind.append(int(l[i]))
    
for i in range(1,len(l),2):
    var.append(float(l[i]))


r1=pd.DataFrame({'constr. emissions':var,'t':ind})
r1=r1.set_index('t')
r1=r1.sort_index()


r=r.join(r1['constr. emissions'])
scc=-1000*r['constr. emissions']/r['constr. accounting']
r['scc']=scc


TT = np.linspace(2000, Time_horizon,100 , dtype = np.int32)



print('Drupp & Hänsel')
print(scc[4])
print(scc[20])


plt.plot(TT[4:21], scc[4:21],'g:', label='Drupp & Hänsel', color='black')