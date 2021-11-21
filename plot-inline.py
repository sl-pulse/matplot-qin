#---Graph plotting script is written inline in data processing function
#——Author: Silvena Lam
import re, calendar, csv, datetime #re is for regex
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #for matching colour of legend and dictionary
from matplotlib.ticker import ScalarFormatter #classes for configuring tick locating and formatting
from datetime import datetime
from datetime import timedelta
from pathlib import Path #Note to self: library Path is capitalized, import path instead of Path would incur error
budget = [ #Change this list manually in code if need arises
180000, #July 2020
360000,	#Aug 2020
520000, #Sep 2020
700000, #Oct 2020
900000, #Nov 2020
110000, #Dec 2020
130000, #Jan 2021
70000, #Feb 2021
100000, #Mar 2021
130000, #Apr 2021
100000, #May 2021
800000, #June 2021
]
#---Dictionary for sale person and responsible region mapping
region_dict = {
"Adam":"Europe",
"Betty":"India & MENA",
"Charlie":"China”,
"David":"Japan",
"Elsa":"ANZ",
"Fiona":"North America"
}
all_stages = ["Prospect", "Quote Sent", "Quote Accepted", "Contract in Negotiation", "Contract Executed"]
selection = ['person', 'region', 'stage', 'product']
selection_text = ["0 for sale person", "1 for region", "2 for deal stage", "3 for product category"]
avail_colours = {'ANZ': '#b3e0ff', 'China':'#A72B2A', 'Europe':'#34657F', 'India & MENA': '#D7D2CB', 'Japan': '#382F2D', 'North America':'#A9794D'}
pct_colours = {'ANZ': '#000000', 'China': '#FFFFFF', 'Europe': '#FFFFFF', 'India & MENA': '#000000', 'Japan': '#FFFFFF', 'North America':'#FFFFFF'}
#---On screen display prettifier
space, star, hash, arrow = " ","*", "#", ">"
space, star, hash, arrow  = (x *80 for x in [space, star, hash, arrow])
blankline = [""]
params = {'legend.fontsize': 'small',
     'legend.labelspacing': 0.35,
     'legend.title_fontsize': 'small',
     'axes.labelsize': 'small',
     'axes.titlesize':'large',
     'axes.titleweight':'bold',
     'axes.titlecolor':'slategrey',
     'font.size': 7,
     'figure.facecolor': 'w',
     'savefig.dpi': 300,
     }
plt.rcParams.update(params)
#Create lists of all months in FY
m1, m2 = list(range(7, 13)), list(range(1, 7))
fy2 = input("\nEnter the Fiscal Year in concern: \n>>")
fy1 = str(int(fy2)-1)
yearmonth = [] #yearmonth list contains all months in this FY in yyyymm format as strings
month_end = [] #month end list contains all month-end-date in this FY in dd-mm-yyyy format as strings
#---Use calendar module to find the last day of each month of this FY
for month in m1: #---Purpose of this is to prevent excel from changing June 2020 to 1/6/2020 by default
    mm = str(month).zfill(2) #Zfill-Fill the strings with leading zeros until they are x characters long:
    yearmonth.append(fy1 + mm)
    findday = calendar.monthrange(int(fy1),int(mm)) #find the last day of the month
    me = str(findday[1]) + "-" + str(mm) + "-" + str(fy1)
    month_end.append(me)
for month in m2:
    mm = str(month).zfill(2)
    yearmonth.append(fy2 + str(month).zfill(2))
    findday = calendar.monthrange(int(fy2),int(mm))
    me = str(findday[1]) + "-" + str(mm) + "-" + str(fy2)
    month_end.append(me)
#---Definition of function StackedBarChart
def StackedBarChart(plot_data, title, patches, fc, gc, graph_name):
#---Convert source data to pandas DataFrame
    df = pd.DataFrame(data=plot_data)
    #---Convert monthend to mmm-yyyy format for axis labels
    for ym in yearmonth:
        df = df.rename(index={ym: datetime.strptime(str(ym), '%Y%m').strftime('%b %Y')})
    fig, axs = plt.subplots(figsize=(7,5))
    x = df.index #RangeIndex(start=0, stop=12, step=1)
    indexes = np.argsort(df.values).T #.T means tranpose, argsort returns the index in value sorted order
    heights = np.sort(df.values).T
    order = -1
    bottoms = heights[::order].cumsum(axis=0) #bottoms contain the start point in y-axis after each value is added
    bottoms = np.insert(bottoms, 0, np.zeros(len(bottoms[0])), axis=0) #add a row contains all zero, numpy.insert(arr, obj, values, axis=None)[source]
    for btms, (idxs, vals) in enumerate(list(zip(indexes, heights))[::order]):
        mps = np.take(np.array(df.columns), idxs)  #df.columns contain the header of the orginal dataframe
        #---np.take, return elememts from array along the mentioned axis and indices.
        #not sure what mps does need to ask on stackflow
        axs.bar(x, height=vals, bottom=bottoms[btms], width=0.65, color=[avail_colours[m] for m in mps])
    axs.set_title(title, y=1.01)
    axs.legend(handles=patches)
    axs.set_facecolor(fc)
    axs.grid(axis='y', color=gc)
    axs.set_axisbelow(True) #put gridline behind the bars instead of in front of them
    axs.set_ylim(bottom=0, top=2200000) #y axis range
    axs.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('US${x:,.0f}')) #convert y-axis values to currency format
    plt.xticks(rotation=45) #rotate label by 45 deg
    fig.subplots_adjust(left=0.145, right=0.949, top=0.863, bottom=0.152)
    fig.savefig(graph_name, transparent=False)
#---End of definition of StackedBarChart
#---Function definition of Monthlyaverage
def MonthlyAverage(sc, yearmonth):
    output = []
    m_total, rwm_total = dict(), dict()
    for ym in yearmonth:
        m_total[ym], rwm_total[ym] = 0, 0
    for rec in sc:
        row_monthly = []
        edyyyy, sdyyyy, edmm, sdmm = int(rec['edyyyy']), int(rec['sdyyyy']), int(rec['edmm']), int(rec['sdmm'])
        for thisym in yearmonth:
            thisyear = thisym[0:4]
            thismonth = thisym[4:]
            if edyyyy < int(thisyear) or (edyyyy == int(thisyear) and edmm < int(thismonth)):#past deals
                income, riskincome = 0, 0
            elif sdyyyy > int(thisyear) or (sdyyyy == int(thisyear) and sdmm > int(thismonth)): #future deals
                income, riskincome = 0, 0
            else:
                income = rec['spc']
                riskincome = rec['rspc']
                m_total[thisym] += income
                rwm_total[thisym] += riskincome
            row_monthly.append(income) #list contains averaged monthly sale from this deal for the entire FY
        row_monthly.append(sum(row_monthly)) #total income from this deal in this FY
        output.append(row_monthly) #add calculated results to final output by row
    global monthly_total, rwmonthly_total
    monthly_total, rwmonthly_total = [], []
    monthly_total = list(m_total.values())
    monthly_total.append(sum(monthly_total)) #monthly total of all deals based on averaged income i.e. vertical total
    rwmonthly_total = list(rwm_total.values())
    rwmonthly_total.append(sum(rwmonthly_total)) #rw-monthly total of all deals based on averaged rw-income
    return output
#End of function by_Monthlyaverage
#---Function definition of MonthlyAveragePrint
def MonthlyAveragePrint(source, m_avg, m_avg_head):
    keep_keys = ['id', 'deal', 'person', 'region', 'stage', 'company', 'product', 'quote', 'potential', 'amt', 'likelihood', 'sd', 'ed']
    count = 0
    m_avg_head.extend(month_end)
    m_avg_head.append("FY Total")
    output = [m_avg_head]
    while count < len(m_avg):
        line = []
        for key, value in source[count].items(): #get key value pairs of each row
            for item in keep_keys: #add value to current row of output if the key is in the keep list
                if item == key:
                    line.append(value)
        line.extend(m_avg[count]) #combine data of this row in the source file with calculated monthly averaged sale amount/income
        output.append(line) #add this row of output to the list for output
        count += 1
    return output
#End of function MonthlyAveragePrint
#---Function defintion of ByFiscalYear
def ByFiscalYear(sc):
    counter, rwsale, sale = (dict(),dict(),dict())
    for year in years_found:
        counter[year], rwsale[year], sale[year] = 0, 0, 0
    for rec in sc:
        if int(rec['sdmm']) < 7:
            counter[rec['sdyyyy']] += 1
            rwsale[rec['sdyyyy']] += rec['rw_amt']
            sale[rec['sdyyyy']] += rec['amt']
        else:
            counter[str(int(rec['sdyyyy'])+1)] += 1
            rwsale[str(int(rec['sdyyyy'])+1)] += rec['rw_amt']
            sale[str(int(rec['sdyyyy'])+1)] += rec['amt']
    ct, rws, s, avg = (["Count"], ["Risk Weighted$"], ["$"], ["Average deal size"])
    for year in years_found:
        ct.append(counter[year])
        rws.append(rwsale[year])
        s.append(sale[year])
        if sale[year] != 0:
            avg.append(sale[year]/counter[year])
        else:
            avg.append(0)
    title = "By Fiscal Year (All Deals)"
    cy_head = years_found[:]
    cy_head.insert(0,"")
    cy_head.append("Total")
    t_ct = sum(ct[1:]) #the first element in list ct is the row header, so we need to slice the list, if we want to get the sum of all elements
    t_s = sum(s[1:]) #t_ct is total count, t_s is total sale amount. Defining as new variables as it wil be reused for calculation of avg size of deal
    ct.append(t_ct)
    rws.append(sum(rws[1:]))
    s.append(t_s)
    avg.append(t_s/t_ct)
    FiscalChart(ct[1:-1], avg[1:-1], t_s, sum(rws[1:-1]), rws[1:-1], s[1:-1], years_found)
    output = [blankline, [title], cy_head, ct, rws, s, avg]
    return output
#---End of function ByFiscalYear
def FiscalChart(count, average, sum, rw_sum, weighted, non_weighted, years_found):
    width = 0.5
    list = [weighted, non_weighted]
    df = pd.DataFrame(data=list, columns=years_found, index=['Risk Weighted US$', 'Revenue US$']).T
    ax = df[['Risk Weighted US$', 'Revenue US$']].plot(kind='bar', width=width, color=['#D7D2CB', '#A9794D'])
    ax.set_title('Revenue View by Fiscal Year', pad=15)
    ax.legend(loc='best')
    ax.grid(axis='y')
    ax.set_axisbelow(True) #put gridline behind the bars instead of in front of them
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('US${x:,.0f}')) #convert y-axis values to currency format
    plt.xticks(rotation=45) #rotate label by 45 deg
    plt.tight_layout(pad=3, h_pad=None, w_pad=None)
    plt.savefig('fiscalyear_view.png')
    return
#---Function definition of ByLikelihood
def ByLikelihood(sc, lp, hp):
    output = []
    low_count, mid_count, high_count = 0, 0, 0
    low_sum, mid_sum, high_sum = 0, 0, 0
    low_rw_sum, mid_rw_sum, high_rw_sum = 0, 0, 0
    for rec in sc:
        if rec['likelihood'] < float(lp):
            low_count += 1
            low_sum += rec['amt']
            low_rw_sum += rec['rw_amt']
        elif rec['likelihood'] > float(hp):
            high_count += 1
            high_sum += rec['amt']
            high_rw_sum += rec['rw_amt']
        else:
            mid_count += 1
            mid_sum += rec['amt']
            mid_rw_sum += rec['rw_amt']
    total_count  = low_count + mid_count + high_count
    total_sum = low_sum + mid_sum + high_sum
    total_rw_sum = low_rw_sum + mid_rw_sum + high_rw_sum
    #---Formatting of data for output
    lper = "<" + str(int(float(lp) * 100)) + "%"
    hper = ">" + str(int(float(hp) * 100)) + "%"
    mper = str(int(float(lp) * 100)) + "-" + str(int(float(hp) * 100)) + "%"
    title = "By Likelihood (All Deals)"
    output = [blankline,[title]]
    output.append(["", lper, mper, hper, "Total"])
    output.append(["Count", low_count, mid_count, high_count, total_count])
    output.append(["Risk Weighted$", low_rw_sum, mid_rw_sum, high_rw_sum, total_rw_sum])
    output.append(["$", low_sum, mid_sum, high_sum, total_sum])
    return output
#---End of function ByLikelihood
#---Function definition of ByDealStage
def ByDealStage(sc):
    count_row, rwm_row, m_row  = ["Count"],["Risk Weighted$"],["$"] #row header
    ds_count, ds_m, ds_rwm = dict(), dict(), dict()
    title = "By Deal Stage (All Deals)"
    for rec in sc:
        stage = rec['stage']
        if stage not in ds_count:
            ds_count[stage] = 1
            ds_rwm[stage] = rec['rw_amt']
            ds_m[stage] = rec['amt']
        else:
            ds_count[stage] += 1
            ds_rwm[stage] += rec['rw_amt']
            ds_m[stage] += rec['amt']
    #---Formatting of data for output
    stages_head = stages_incl[:]
    stages_head.insert(0, "")
    for stage in stages_incl:
        count_row.append(ds_count[stage])
        rwm_row.append(ds_rwm[stage])
        m_row.append(ds_m[stage])
    output = [blankline, [title], stages_head, count_row, rwm_row, m_row]
    return output
#---End of function ByDealStage
#---Function definition of RevnueChart (plot a revenue view bar/line-plot Charts)
def RevenueChart(mt, rwt, budget, yearmonth):
    width = 0.85
    column_name =[]
    for ym in yearmonth:
        column_name.append(datetime.strptime(str(ym), '%Y%m').strftime('%b %Y'))
    list = [rwt[:-1], mt[:-1], budget]
    df1 = pd.DataFrame(data=list, columns=column_name, index=['Risk Weighted US$', 'Revenue US$', 'Budget US$']).T
    df1[['Budget US$']].plot(linestyle="-", color='#A72B2A', linewidth=2)
    ax = plt.gca()
    df1[['Risk Weighted US$', 'Revenue US$']].plot(kind='bar', width=width, ax=ax, color=['#D7D2CB', '#A9794D'])
    ax.set_title('Revenue View VS Budget', pad=15)
    ax.legend(loc='best')
    ax.grid(axis='y')
    ax.set_axisbelow(True) #put gridline behind the bars instead of in front of them
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('US${x:,.0f}')) #convert y-axis values to currency format
    plt.xticks(rotation=45) #rotate label by 45 deg
    plt.tight_layout(pad=3, h_pad=None, w_pad=None)
    plt.savefig('revenue_view.png')
    return
#---End of function definition of RevnueChart
#---Function definition of Revenue
def Revenue(mt, rwmt, rev_head):
    title = "Revenue View (Averaged Monthly Income)"
    rev_head.insert(0, "") #add space as first element of header
    rev_head.append("Total") #add total as last element of header
    rwmt.insert(0, "Risk Weighted$:")
    mt.insert(0, "$")
    output = [blankline, [title], rev_head, rwmt, mt]
    return output
#----End of function Revenue
#---Function definition of ByCloseDate
def ByCloseDate(sc, cm_head, yearmonth):
    counter, rwsale, sale = (dict(), dict(), dict())
    for ym in yearmonth:
        counter[ym], rwsale[ym], sale[ym] = 0, 0, 0
        for rec in sc:
            if rec['sdym'] == ym:
                counter[ym] += 1
                rwsale[ym] += rec['rw_amt']
                sale[ym] += rec['amt']
    #---Formatting of data for output
    ct, rws, s, avg = (["Count"], ["Risk Weighted$"], ["$"], ["Average deal size"])
    for ym in yearmonth:
        ct.append(counter[ym])
        rws.append(rwsale[ym])
        s.append(sale[ym])
        if sale[ym] != 0:
            avg.append(sale[ym]/counter[ym])
        else:
            avg.append(0)
    cm_head.insert(0, "")
    cm_head.append("FY Total")
    title = "By Deal Close Date"
    t_ct = sum(ct[1:]) #the first element in list ct is the row header, so we need to slice the list, if we want to get the sum of all elements
    t_s = sum(s[1:]) #defined these two sums as separate variables as it will be called upon again for calculation of the average
    ct.append(t_ct)
    rws.append(sum(rws[1:]))
    s.append(t_s)
    avg.append(t_s/t_ct)
    output = [blankline, [title], cm_head, ct, rws, s, avg]
    return output
#---End of function ByCloseDate
#---Function definition of ByCMstage
def ByCMStage(sc, head1, cms_summary, cms_rwsummary, yearmonth):
    title = "By Deal Stage (Deal Close Date in this FY)"
    deal_stage = dict()
    rw_deal_stage = dict()
    for stage in stages_incl: #by deal stage
        deal_stage[stage] = dict()
        rw_deal_stage[stage] = dict()
        for ym in yearmonth: #dictionaries defined in chronological date order
            deal_stage[stage][ym], rw_deal_stage[stage][ym] = 0, 0
    for rec in sc:
        this_sdym = rec['sdym'] #sdym in string not int
        if this_sdym in yearmonth: #we only care about deals that will close in this FY
            this_stage = rec['stage']
            deal_stage[this_stage][this_sdym] += rec['amt']
            rw_deal_stage[this_stage][this_sdym] += rec['rw_amt']
    #---Formatting of data for output
    head1.append("FY Total") #head contains list of column headers
    head1.insert(0, "Non-Weighted")
    output = [blankline, [title], head1]
    for stage in stages_incl:
        temp1 = list(deal_stage[stage].values())
        temp1.append(sum(temp1)) #FY total of this deal-stage
        temp1.insert(0, stage) #Row title
        output.append(temp1)
    cms_summary.insert(0, "Total")
    output.append(cms_summary)
    head2 = head1[:]
    head2[0] = "Risk-Weighted"
    output.extend([blankline, head2])
    for stage in stages_incl:
        temp2 = list(rw_deal_stage[stage].values()) #since list is organised in chronological date orders so we can simple convert values of the dictionary to list
        temp2.append(sum(temp2))
        temp2.insert(0, stage)
        output.append(temp2)
    cms_rwsummary.insert(0, "Total")
    output.append(cms_rwsummary)
    return output
#---End of function ByCMStage
#---Function defintion of BySalePerson
def BySalePerson(sc, head1, sp_summary, sp_rwsummary, yearmonth):
    rep_ym, rep_rw_ym  = dict(), dict()
    for rep in sale_rep:
        rep_ym[rep], rep_rw_ym[rep] = dict(), dict()
        for ym in yearmonth:
            rep_ym[rep][ym], rep_rw_ym[rep][ym] = 0, 0
    for rec in sc:
        this_sdym = rec['sdym']
        if this_sdym in yearmonth:
            rep_ym[rec['person']][this_sdym] += rec['amt']
            rep_rw_ym[rec['person']][this_sdym] += rec['rw_amt']
    #---Formatting of data for output
    title = "By Owner (Deal Close Date in this FY)"
    head1.insert(0, "Non-Weighted")
    head1.extend(["FY Total", "%"])
    output = [blankline, [title], head1]
    fy_total = sp_summary[len(sp_summary)-1]
    for person, pairs in rep_ym.items(): #key of pairs = month | value of pairs = accmulated sale amount in this month
        temp1 = list(pairs.values())
        person_total = sum(temp1)
        temp1.insert(0, person)
        temp1.extend([person_total, person_total/fy_total])
        output.append(temp1)
    sp_summary.insert(0, "Total")
    sp_summary.append(1)
    output.append(sp_summary)
    head2 = head1[:]
    head2[0] = "Risk-Weighted"
    output.extend([blankline, head2])
    fy_rwtotal = sp_rwsummary[len(sp_rwsummary)-1]
    for person, pairs in rep_rw_ym.items():
        temp2 = list(pairs.values())
        person_total = sum(temp2)
        temp2.insert(0, person)
        temp2.extend([person_total, person_total/fy_rwtotal])
        output.append(temp2)
    sp_rwsummary.insert(0, "Total")
    sp_rwsummary.append(1)
    output.append(sp_rwsummary)
    return output
#End of function by_Saleperson
#---Function Definition of ByRegion
def ByRegion(sc, head1, re_summary, re_rwsummary, yearmonth):
    reg_ym, reg_rwym = dict(), dict()
    all_regions = list(set(region_dict.values())) #get all unique values from dictionary
    all_regions.sort()
    for region in all_regions:
        reg_ym[region], reg_rwym[region] = dict(), dict()
        for ym in yearmonth:
            reg_ym[region][ym], reg_rwym[region][ym] = 0, 0
    for rec in sc:
        this_region = rec['region']
        this_sdym = rec['sdym']
        if this_sdym in yearmonth:
            reg_ym[this_region][this_sdym] += rec['amt']
            reg_rwym[this_region][this_sdym] += rec['rw_amt']
    #---Formatting of data for output
    title = "By Region (Deal Close Date in this FY)"
    head1.insert(0, "Non-Weighted")
    head1.extend(["FY Total", "%"])
    output = [blankline, [title], head1]
    fy_total = re_summary[len(re_summary)-1]
    for region, pairs in reg_ym.items(): #sorted seems to have changed it to a list?? need to ask on Stackexchange
        temp1 = (list(pairs.values())) #level 1 key is region, level 2 key is yearmonth (key of pairs), value in pairs is accmulated sum
        region_total = sum(temp1)
        temp1.extend([region_total, region_total/fy_total])
        temp1.insert(0, region)
        output.append(temp1)
    re_summary.insert(0, "Total")
    re_summary.append(1)
    output.append(re_summary)
    head2 = head1[:]
    head2[0] = "Risk-Weighted"
    output.extend([blankline, head2])
    fy_rwtotal = re_rwsummary[len(re_rwsummary)-1]
    for region, pairs in reg_rwym.items():
        temp2 = (list(pairs.values()))
        region_total = sum(temp2)
        temp2.extend([region_total, region_total/fy_rwtotal])
        temp2.insert(0, region)
        output.append(temp2)
    re_rwsummary.insert(0, "Total")
    re_rwsummary.append(1)
    output.append(re_rwsummary)
    #---Define variables for graph
    #---Match Colours and Labels based on dictionary in list order (which is alphatical)
    patches = []
    for region in all_regions:
        patches.append(matplotlib.patches.Patch(color = avail_colours[region], label = region))
    #---Convert non-weighted list to single level dictionary
    title_rw = 'FY '+ fy2 +' revenue by deal close month & region (Risk-Weighted)'
    fn2 = 'region_risk_weight_stacked_bar.png'
    #---Convert weighted list to single level dictionary
    title_normal = 'FY '+ fy2 +' revenue by deal close month & region (Non-Weighted)'
    fn1 = 'region_risk_non_weight_stacked_bar.png'
    StackedBarChart(reg_ym, title_normal, patches, '#FFFFFF', '#D8D8D8', fn1)
    StackedBarChart(reg_rwym, title_rw, patches, '#EEE4DB', '#FFFFFF', fn2)
    return output
#End of function ByRegion
#---Function defintion of ByLevels
def ByLevels(sc, crit1, crit2, stages_incl):
    res = dict()
    rw_res = dict()
    count = dict()
    head = ["", "Count","Risk Weighted US$", "Revenue US$"]
    title = "By "+ crit1.capitalize() + " & " + crit2.capitalize()
    output = [blankline, [title]]
    if crit1 == "stage": #if stage is a criteria, use stage_incl instead so results will be sorted in desired/customised order
        unique_crit1 = stages_incl
        unique_crit2 = set()
        for row in sc:
            unique_crit2.add(row[crit2])
        unique_crit2 = sorted(unique_crit2) #cannot use sort() because, set is unordered
    elif crit2 == "stage":
        unique_crit2 = stages_incl
        unique_crit1 = set()
        for row in sc:
            unique_crit1.add(row[crit1])
        unique_crit1 = sorted(unique_crit1)
    else:
        unique_crit1, unique_crit2 = set(), set() #define as set instead of list
        for row in sc:
            unique_crit1.add(row[crit1]) #use add() instead of append() when dealing with set
            unique_crit2.add(row[crit2])
        unique_crit1 = sorted(unique_crit1)
        unique_crit2 = sorted(unique_crit2)
    for uc1 in unique_crit1:
        res[uc1], rw_res[uc1], count[uc1]  = dict(), dict(), dict()
        for uc2 in unique_crit2:
            res[uc1][uc2], rw_res[uc1][uc2], count[uc1][uc2] = 0, 0, 0
    for row in sc:
        res[row[crit1]][row[crit2]] += row['amt']
        rw_res[row[crit1]][row[crit2]] += row['rw_amt']
        count[row[crit1]][row[crit2]] +=1
    #---Formatting of data for output
    for c1 in unique_crit1: #---Add Header & Sub-heading sum
        output.append(head)
        temp = [c1, sum(count[c1].values()), sum(rw_res[c1].values()), sum(res[c1].values())]
        output.append(temp)
        for c2 in unique_crit2:
            temp = [c2, count[c1][c2], rw_res[c1][c2], res[c1][c2]]
            output.append(temp)
    #Plot Charts
    fig, axs = plt.subplots(2, 2, figsize=(8, 6.5))
    axs = axs.ravel()
    count = 0
    for c1 in unique_crit1:
        labels, colours, sizes = [], [], []
        for c2 in unique_crit2:
            if res[c1][c2] != 0:
                labels.append(c2)
                sizes.append(res[c1][c2])
                if crit2 == "region": #need a separate list of colours for when criteria isn't region
                    colours.append(avail_colours[c2])
        sizes, labels, colours= zip(*sorted(zip(sizes, labels, colours), reverse = True)) #Plot Largest Slice of Pie first
        explode = [0.01 for i in range(0, len(sizes))]
        ttl = c1 + " \n$" + str('{:,}'.format(int(sum(res[c1].values()))))
        #axs[count].pie(sizes, labels=labels, explode=explode, shadow=False,
        #autopct='%1.1f%%', colors = colours, pctdistance = 0.65, startangle=180, labeldistance = 1.1, radius = 1, textprops={'color':"w"})
        wedges, texts, autotexts = axs[count].pie(sizes, labels=labels, explode=explode, shadow=False,
        autopct='%1.1f%%', colors = colours, pctdistance = 0.7, startangle=180, labeldistance = 1.1, radius = 1)
        for x in range(len(texts)):
            text = str(texts[x])
            for region in unique_crit2:
                if region in text:
                    autotexts[x].set_color(pct_colours[region])
        axs[count].set_title(ttl, y=0.98)
        axs[count].set_position([0.15, 0.575, 0.38, 0.38])
        count += 1
    handles, labels = axs[2].get_legend_handles_labels() #Cheat: Use [2] because we know that one covers all regions. note: Learned after writing this Patch would be better
    fig.legend(handles, labels, loc = 'center', bbox_to_anchor=(0.361, 0.007, 0.3, 1), ncol=3, mode="expand", borderaxespad=0., fancybox=True, shadow=True)
    fig.subplots_adjust(hspace = .285, wspace=.307, top=0.93, bottom=0.012)
    graph_name = crit1 + "_" + crit2 +".png"
    fig.savefig(graph_name, transparent=False)
    return output
#---End of function ByLevels
filetype = ".csv"
fn =  input("\nType in the name of your source file (no file extension needed): \n>>")
filename = fn+filetype
#---Open csv source file
with open(filename, encoding='ISO-8859-1') as csvfile: #Use encoding ISO-8859-1 to avoid problems caused by accented characters i.e. French/Spanish brands
    readCSV = csv.reader(csvfile, delimiter=',')
    headers = next(readCSV)
    head = headers[:]
    head.insert(3, "Region") #insert at position 3 and shift original elements right
#---Initiate variables/lists for storing data from source file
    sale_rec, sale_rep, stages_found, years_found = [], [], [], []
#---Read input begins
    for row in readCSV:
        line = dict()
        line['id'] = row[0]
        line['deal'] = row[1]
        #----Formatting of names - Sort out random casing of names i.e. some names are all cap or not capitalised
        temp_person = row[2].split() #split name by whitespace and store in list. i.e [First Name, Last Name]
        person = "" #initate rep
        for per in temp_person: #get element in list and capitalize each
            person += " " + per.capitalize()
        person = person.strip() #get rid of leading and trailing whitespace
        #---End of Formatting of name
        line['person'] = person
        if person not in sale_rep:
            sale_rep.append(person) #create a list of sale persons for later use
        line['region'] = region_dict[person]
        line['stage'] = row[3]
        if row[3] not in stages_found: #create a list of stages found in source file
            stages_found.append(row[3])
        line['company'] = row[4]
        line['product'] = row[5]
        line['quote'] = row[6]
        line['potential'] = row[7]
        #---Formatting of Sale Amount - To avoid halt of operation due to division performed on empty value
        if not row[8]:
            vol = float(0)
        else:
            vol = float(row[8])
        #---End of formatting of sale amount
        line['amt'] = vol
        #---Format likelihood - To sort input as float. i.e. get rid of % sign and to sort to empty value to avoid operating on empty value
        like = row[9] #temp value
        non_decimal = re.compile(r'[^\d.]+') #regex to get rid of any non numerical values. decimal place is allowed
        #Alternative solution: filter( lambda x: x in '0123456789.', s ) but regex is faster
        likeli = non_decimal.sub('', like) #syntax: re.sub(pattern,repl,string)| Replace non decimal char with empty string
        if not likeli:
            likeli = 38.88 #Set dummy percentage 38.88% if value non-specified in source
        likelihood = float(likeli)/100
        #---End of formatting of likelihood
        line['rw_amt']= vol * likelihood
        line['likelihood'] = likelihood
        #---Formatting of close deal date. Stored as string with date & time
        if '-' in row[10]:
            sd_input = datetime.strptime(row[10], '%Y-%m-%d %H:%M') #convert string to time for in app use
        elif '/' in row[10]:
            sd_input = datetime.strptime(row[10], '%d/%m/%Y %H:%M') #yet the script doesn't tackle if date is organised as m/d/y
        sd = sd_input.strftime('%d-%b-%Y') #change format of date to dd-mmm-yyyy for output
        sd_temp = sd_input.strftime('%d-%m-%Y') #convert date to a unified format before slicing
        sdyyyy = sd_temp[6:10]
        sdmm = sd_temp[3:5]
        sdym = str(sdyyyy) + str(sdmm)
        if int(sdmm) > 6:
            if str(int(sdyyyy)+1) not in years_found: #years_found in used in by FiscalYear type of summary
                years_found.append(str(int(sdyyyy)+1))
        else:
            if sdyyyy not in years_found:
                years_found.append(sdyyyy)
        #---End of formatting of close deal date
        line['sd'], line['sdyyyy'], line['sdmm'], line['sdym'] = sd, sdyyyy, sdmm, sdym
        #---Handling of empty date (Exception Catcher) by using dummy date if field is empty
        #---Formatting of contract end datetime
        if not row[11]: #if deal close date is empty, use dummy date
            ed_input = datetime.strptime('2000-01-01', '%Y-%m-%d')
            edyyyy = "2000"
            edmm = "01"
        else:
            if '-' in row[11]:
                ed_input = datetime.strptime(row[11], '%Y-%m-%d')
            elif '/' in row[11]:
                ed_input = datetime.strptime(row[11], '%d/%m/%Y')
            ed_temp = ed_input.strftime('%d-%m-%Y')
            edyyyy = ed_temp[6:10]
            edmm = ed_temp[3:5]
        ed = ed_input.strftime('%d-%b-%Y') #change format of date to dd-mmm-yyyy for output
        edym = str(edyyyy) + str(edmm)
        #---End of formatting of contract end date
        line['ed'], line['edyyyy'], line['edmm'], line['edym'] = ed, edyyyy, edmm, edym
        #---Calculation of execution period of contract, amound and risk weighted amount per month
        yeardiff = int(edyyyy) - int(sdyyyy)
        monthdiff = int(edmm) - int(sdmm)
        if yeardiff > 0:
            timediff = yeardiff * 12 + monthdiff + 1
        elif yeardiff < 0: #exception catcher: negative year caused by input error in source data
            timediff = 0
        elif monthdiff >= 0: #deal starts and ends on the same year, time diff = diff in months
            timediff = monthdiff + 1
        else: #exception catcher: negative month diff i.e. end date is earlier month of same year than start date
            timediff=0
        if vol != 0 and timediff != 0:
            spc = vol / timediff #sale income per delivery cycle
            rspc = spc * likelihood #risk-weighted sale income per delivery cycle
        else:
            spc, rspc = 0, 0 #0/0 is infinitive, so override by setting result to zero to avoid error
        line['spc']= spc
        line['rspc'] = rspc
        #---End of calculaton
        sale_rec.append(line)
    sale_rep.sort() #sort items in list alphatically
    stages_incl = sorted(set(all_stages) & set(stages_found), key = all_stages.index) #compare the two sets (set has no order) and use index in list to sort compared result
    years_found.sort()
print(f"\n{star[0:6]}{space[0:4]}{star[0:6]}{space[0:4]}{star[0:6]}{space[0:4]}OK, found your file.{space[0:4]}{star[0:6]}{space[0:4]}{star[0:6]}{space[0:4]}{star[0:6]}")
#---End of reading & processing of source file
#---Ask for the list of reports to run/print
all_report = input(
"\nWould you like to run FULL SET of standard reports?\n\n"
"(Standard Reports Use 50% and 80% for Likelihood Summary & Deal-Stage/Region for Two-Tier Analysis)\n\n"
"Type 'y' to run full standard reports, 'n' to customise reports\n>>")
if all_report == "y":
    reports = ['m_avg_print', 'by_close_year','by_like','by_stage','rev_view','c_date_print','cm_stage', 'cm_region', 'decked']
    for report in reports:
        globals() [report] = "y" #locals()[o+str(p)] = [stage]
    lp, hp = 0.5, 0.8
    lev1, lev2 = 2, 1
    cm_person = 'n'
if all_report != "y":
    print(f"\n{hash}\n\nType 'y' if the answer is yes to the below questions\nOR press any other key to skip\n\n")
    m_avg_print = input("Would you like to generate a monthly averaged (MA) sale income report?\n>>")
    by_close_year = input("\nWould you like to include a by deal-close fiscal year (AD) report?\n>>")
    by_like = input("\nWould you like to include a by likelihood summary (AD) in your report?\n>>")
    if by_like == "y":
        lp=input("\n\tPlease specific a lower limit for likelihood summary i.e. type in 0.5 for 50% or less\n\t>>")
        hp=input("\n\tPlease specific a upper limit for likelihood summary i.e. type in 0.75 for 75% or more\n\t>>")
    by_stage = input("\nWould you like to include a by deal stage summary (AD) in your report?\n>>")
    rev_view = input("\nWould you like to include a revenue view for the FY in comcerm(MA)?\n>>")
    c_date_print = input("\nWould you like to include a monthly summary based on deal-close date (CDM)?\n>>")
    cm_stage = input("\nWould you like to include a by-deal-stage summary (CDM)?\n>>")
    cm_person = input("\nWould you like to include a by-deal-owner summary (CDM)?\n>>")
    cm_region = input("\nWould you like to include by-region summary (CDM)?\n>>")
    decked = input("\nWould you like to include a two-tiered analysis (AD)?\n>>")
    if decked == "y":
        print("\tPlease select the first criteria:\n\tEnter the corresponding number to make selection")
        for sel in selection_text:
            print(f"\t{sel}")
        lev1 = int(input("\t>>"))
        print("\n\tPlease select the second criteria:\n\tEnter the corresponding number to make selection")
        nst = selection_text[:]
        nst.pop(lev1)
        for sel in nst:
            print(f"\t{sel}")
        lev2 = int(input("\t>>"))
#---Done asking questions
start = datetime.now()#----Start timer
print (f"\n{arrow[0:7]}Calculation Begins{arrow[0:55]}")
#---Run functions based on user selection
m_avg = MonthlyAverage(sale_rec, yearmonth)
c_date_output = ByCloseDate(sale_rec, month_end.copy(), yearmonth)
cdm_total = c_date_output[5][1:]
rw_cdm_total = c_date_output[4][1:]
if m_avg_print == "y":
    import copy #need to import copy in order to call the deepcopy function
    m_avg_output = MonthlyAveragePrint(copy.deepcopy(sale_rec), m_avg, head.copy()) #head is the headers of the source file
if  by_close_year == "y":
    c_year_output = ByFiscalYear(sale_rec)
if rev_view == "y":
    rev_output = Revenue(monthly_total.copy(), rwmonthly_total.copy(), month_end.copy())
    RevenueChart(monthly_total.copy(), rwmonthly_total.copy(), budget, yearmonth)
if by_like == "y":
    like_output = ByLikelihood(sale_rec, lp, hp)
if by_stage == "y":
    stage_output = ByDealStage(sale_rec)
if cm_stage == "y":
    stage_by_month = ByCMStage(sale_rec, month_end.copy(),cdm_total.copy(), rw_cdm_total.copy(), yearmonth)
if cm_person == "y":
    owner_by_month = BySalePerson(sale_rec, month_end.copy(),cdm_total.copy(), rw_cdm_total.copy(), yearmonth)
if cm_region == "y":
    region_by_month = ByRegion(sale_rec, month_end.copy(),cdm_total.copy(), rw_cdm_total.copy(), yearmonth)
if 'lev1' and 'lev2' in locals(): #meaning if lev1 and lev2 is not blank
    by_two_levels = ByLevels(sale_rec, selection[lev1], selection[lev2], stages_incl)
#---Done running functions
print (f"\n\t\t\tRows Processed: {len(sale_rec)}\n\t\tProcessing Time: {datetime.now()-start}")
print (f"\n{arrow[0:51]}Calculation Completed{arrow[0:8]}")
outputting=input("\nPlease enter a new filename if you would like to save results: \n>>")
#---Write result to files
with open((outputting+filetype), 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    if 'm_avg_output' in locals():
        writer.writerows(m_avg_output)
    if 'c_year_output' in locals():
        writer.writerows(c_year_output)
    if 'like_output' in locals():
        writer.writerows(like_output)
    if 'stage_output' in locals():
        writer.writerows(stage_output)
    if 'rev_output' in locals():
        writer.writerows(rev_output)
    if c_date_print == "y":
        writer.writerows(c_date_output)
    if 'stage_by_month' in locals():
        writer.writerows(stage_by_month)
    if 'owner_by_month' in locals():
        writer.writerows(owner_by_month)
    if 'region_by_month' in locals():
        writer.writerows(region_by_month)
    if 'by_two_levels' in locals():
        writer.writerows(by_two_levels)
#---End of writing to output file
print(f"\n{hash}")
if not outputting:
    print(f"\n{space[0:32]}See you next time!\n")
else:
    print (f"\n{space[0:6]}Your output file can be found in {Path(__file__).parent.absolute()}/{outputting+filetype}\n")
print (f"{hash[0:35]}All Done!{hash[0:36]}\n")
