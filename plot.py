#---Graph plotting script with classes
#——Author: Silvena Lam
import calendar
import datetime
import re #re = regex
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from dateutil import parser
from pathlib import Path #Path is capitalized, import path instead of Path would raise error
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
region_dict = {
"Adam":"Europe",
"Betty":"India & MENA",
"Charlie":"China”,
"David":"Japan",
"Elsa":"ANZ",
"Fiona":"North America"
}
all_stages = ["Prospect", "Quote Sent", "Quote Accepted", "Contract in Negotiation", "Contract Executed"]
#---On screen display prettifier
space, star, hash, arrow = " ","*", "#", ">"
space, star, hash, arrow  = (x *80 for x in [space, star, hash, arrow])
blankline = [""]
class PlotGraph:
    region_colours = {'ANZ': '#b3e0ff', 'China':'#A72B2A', 'Europe':'#34657F', 'India & MENA': '#D7D2CB',
                    'Japan': '#382F2D', 'North America':'#A9794D'}
    reg_pct_colours = {'ANZ': '#000000', 'China': '#FFFFFF', 'Europe': '#FFFFFF', 'India & MENA': '#000000',
                    'Japan': '#FFFFFF', 'North America':'#FFFFFF'}
    colour_list = ['#D7D2CB', '#A9794D', '#A72B2A']
    params = {'legend.fontsize': 'small',
         'legend.labelspacing': 0.35,
         'legend.title_fontsize': 'small',
         'axes.labelsize': 'small',
         'axes.titlesize':'large',
         'axes.titleweight':'bold',
         'axes.titlecolor':'slategrey',
         'font.size': 7,
         'figure.facecolor': 'w',
         'savefig.dpi': 300
         }
    plt.rcParams.update(params)
    def __init__(self):
        pass
    def bar_chart(self, header, list, ind, title, figname):
        bar_label_color = ['#696662', '#6e4f32', '#781f1e']
        df =  pd.DataFrame(data=list, columns=header, index=ind).T
        ax = df[ind].plot(kind='bar', width=0.5, color=self.colour_list)
        for ind, row in enumerate(list): #offset to solve overlapping problem with labels
            for index, value in enumerate(row):
                val = '${:,.0f}'.format(value) #
                if ind% 2 == 0:
                    offset = len(val)*-0.041
                else:
                    offset = len(val)*-0.008 if len(val) > 8 else len(val)*0.002
                font = {'size': 'small', 'color': bar_label_color[ind]}
                ax.text(index + offset, value + 200000, '${:,.0f}'.format(value), fontdict = font)
        ax.set_title(title, pad=15)
        ax.legend(loc='best')
        ax.grid(axis='y')
        ax.set_axisbelow(True) #put gridline behind the bars instead of in front of them
        max_val = max([max(row) for row in list]) #get max value in list for ylim calculation
        mv_digit = len(str(int(max_val))) #get number of digits of the max value
        ax.set_ylim(bottom=0, top = (round(max_val, -mv_digit + 2) + 10**(mv_digit-2)*2))
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('US${x:,.0f}')) #convert y-axis values to currency format
        plt.xticks(rotation=45) #rotate label by 45 deg
        plt.tight_layout(pad=3, h_pad=None, w_pad=None)
        plt.savefig(figname)
        plt.close()
        return
    def bar_and_line(self, header, line_list, bar_list, ind, title, figname):
        df1 = pd.DataFrame(data=line_list, columns=header, index=ind[2:3]).T #pandas dataframe
        df2 = pd.DataFrame(data=bar_list, columns=header, index=ind[0:2]).T
        df1[ind[2:3]].plot(linestyle="-", color=self.colour_list[2], linewidth=2) #line plot
        ax = plt.gca() #get current axis and set as ax
        df2[ind[0:2]].plot(kind='bar', width=0.85, ax=ax, color=self.colour_list) #2 columns barchart on axes ax
        ax.set_title(title, pad=15)
        ax.legend(loc='best')
        ax.grid(axis='y') #show gridline for y axis
        ax.set_axisbelow(True) #put gridline behind the bars instead of in front of them
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('US${x:,.0f}')) #convert y-axis values to currency format
        plt.xticks(rotation=45) #rotate label by 45 deg
        plt.tight_layout(pad=3, h_pad=None, w_pad=None)
        plt.savefig(figname)
        plt.close()
        return
    def stacked_bar(self, df, title, crtieria, fc, gc, graph_name):
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
            axs.bar(x, height=vals, bottom=bottoms[btms], width=0.65, color=[self.region_colours[m] for m in mps])
        axs.set_title(title, y=1.01)
        if crtieria == 'region':
            patches = []
            for region in list(df_rw.index):
                patches.append(matplotlib.patches.Patch(color = self.region_colours[region], label = region))
            axs.legend(handles=patches)
        axs.set_facecolor(fc)
        axs.grid(axis='y', color=gc)
        axs.set_axisbelow(True) #put gridline behind the bars instead of in front of them
        axs.set_ylim(bottom=0, top=2200000) #y axis range
        axs.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('US${x:,.0f}')) #convert y-axis values to currency format
        plt.xticks(rotation=45)
        fig.subplots_adjust(left=0.145, right=0.949, top=0.863, bottom=0.152)
        fig.savefig(graph_name, transparent=False)
        plt.close(fig)
    def pie_chart(self, data, div, criteria, figname):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6.5))
        axs = axs.ravel()
        start, end = 0, div
        patch_list =[data[i][0] for i in range (start + 2, end)]
        for x in range(int(len(data)/div)):
            labels, sizes = [], []
            for y in range (start + 2, end):
                if data[y][-1] != 0:
                    sizes.append(data[y][-1])
                    labels.append(data[y][0])
            if criteria == 'region':
                colours = [self.region_colours[label] for label in labels]
            else:
                colour = None
            sizes, labels, colours = zip(*sorted(zip(sizes, labels, colours), reverse = True)) #Plot the largest slice of pie first
            sub_title = data[start+1][0]+ " \n$" + str('{:,}'.format(data[start+1][-1]))
            explode = [0.01 for i in range(len(sizes))]
            wedges, texts, autotexts = axs[x].pie(sizes, labels=labels, explode=explode, shadow=False,
            autopct='%1.1f%%', colors = colours, pctdistance = 0.7, startangle=180, labeldistance = 1.1, radius = 1)
            if criteria == 'region':
                for z in range(len(texts)):
                    for reg in labels:
                        if reg in str(texts[z]):
                            autotexts[z].set_color(self.reg_pct_colours[reg])
            axs[x].set_title(sub_title, y=0.98)
            axs[x].set_position([0.15, 0.575, 0.38, 0.38])
            start += div
            end += div
        if criteria == 'region':
            patches = []
            patch_list.sort()
            for reg in patch_list:
                patches.append(matplotlib.patches.Patch(color = self.region_colours[reg], label = reg))
            fig.legend(patches, patch_list, loc='center', bbox_to_anchor=(0.361, 0.007, 0.3, 1), ncol=3,
            mode="expand", borderaxespad=0., fancybox=True, shadow=True)
        fig.subplots_adjust(hspace=.285, wspace=.307, top=0.93, bottom=0.012)
        fig.savefig(figname, transparent=False)
        plt.close(fig)
        return
    def donut_chart(self, data, div, figname):
        from matplotlib import cm
        start, end = 0, div
        crit1_labels = []
        crit1_sizes = []
        crit2_labels =[]
        crit2_sizes = []
        cmap_list = ['a', 'b', 'c', 'd', 'e', 'f']
        a, b, c, d, e, f=[plt.cm.Purples, plt.cm.Blues, plt.cm.Greens, plt.cm.Reds, plt.cm.Greys, plt.cm.Oranges]
        crit1_colors = []
        crit2_colors = []
        count = 0
        for x in range(int(len(data)/div)):
            inital_grad = 1
            crit1_labels.append(data[start+1][0])
            crit1_sizes.append(data[start+1][-1])
            string = cmap_list[count] + "(1)"
            subgroup_labels, subgroup_sizes = [], []
            for y in range (start + 2, end):
                if data[y][-1] != 0:
                    subgroup_labels.append(data[y][0])
                    subgroup_sizes.append(data[y][-1])
            subgroup_sizes, subgroup_labels = zip(*sorted(zip(subgroup_sizes, subgroup_labels)))
            crit2_labels.extend(subgroup_labels)
            crit2_sizes.extend(subgroup_sizes)
            this_string = cmap_list[count] + "(" + str(inital_grad-0.1) + ")" #crit2_colors.append(locals()[this_string])
            count += 1
            start += div
            end += div
        crit1_colors = [a(0.7), b(0.7), c(0.7), d(0.7)]
        crit2_colors = [a(0.1), a(0.2), a(0.3), a(0.4), a(0.5),
        b(0.1) , b(0.2), b(0.3), b(0.4), b(0.5),
        c(0.1), c(0.2), c(0.3), c(0.4), c(0.5), c(0.6),
        d(0.1), d(0.2), d(0.3)]
        fig, ax = plt.subplots()
        ax.axis('equal')
        outer, outer_text = ax.pie(crit2_sizes, radius=1.3, labels=crit2_labels, colors=crit2_colors, startangle=0, rotatelabels=True)
        plt.setp( outer, width=0.3, edgecolor='white')
        inner, inner_text, inner_auto = ax.pie(crit1_sizes, radius=1, labels=crit1_labels, labeldistance=0.6, colors=crit1_colors, autopct='%1.1f%%', startangle=0)
        plt.setp( inner, width=0.5, edgecolor='white')
        for x in range(len(inner_auto)):
            percentage = re.findall("\d*\.\d+%|\d+%", str(inner_auto[x]))[0]
            temp = crit1_labels[x].split()
            separate = '\n'
            n_label = separate.join(temp)
            n_label = n_label.rstrip()
            new_label = n_label + '\n' + percentage
            inner_text[x].set_text(new_label)
            inner_auto[x].set_text('')
        for wedge, txt in zip(inner, inner_text):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = wedge.r * 0.75 * np.cos(angle * np.pi / 180)
            y = wedge.r * 0.75 * np.sin(angle * np.pi / 180)
            txt.set_position((x, y))
            txt.set_va("center")
            txt.set_ha("center")
        for wedge, txt in zip(outer, outer_text):
            diff_theta = wedge.theta2 - wedge.theta1
            angle = (wedge.theta2 + wedge.theta1) / 2.
            x = wedge.r * 1 * np.cos(angle * np.pi / 180) #180 *np.cos(angel*0.0174)
            y = wedge.r * 1 * np.sin(angle * np.pi / 180)
            if diff_theta > 1 and diff_theta <= 2:
                txt.set_position((x+0.05, y-0.13))
            if diff_theta <= 1:
                txt.set_position((x+0.024, y-0.16))
        plt.margins(0,0)
        #plt.show()
        return
class DateHandler:
    def __init__(self):
        pass
    def findformat(self, str):
        date = parser.parse(str)
        return date
    def str2date(self, str):
        date = datetime.strptime(str, '%Y-%m-%d')
        return date
    def deconstruct(self, date):
        str = date.strftime('%Y-%m-%d') #das short for date as string
        date = self.str2date(str) #to get rid of the original time
        day = str[8:10]
        month = str[5:7]
        year = str[0:4]
        graph_date = date.strftime('%b %Y')
        res = [date, str, day, month, year, graph_date]
        return res
    def date_header(self, date):
        header = date.strftime('%d-%b-%Y')
        return header
    def construct(self, option, year, month, day=0):
        if option == 'first': #construct a date which is the first day of the month
            day = 1
        elif option == 'last': #construct a date which is the last day of the month
            day = calendar.monthrange(year, month)[1]
        date = datetime(year, month, day)
        return date
    def year_range(self, strings):
        for index, str in enumerate(strings):
            date = self.str2date(str)
            decon = self.deconstruct(date)
            if index == 0:
                first_fy = int(decon[4]) if int(decon[3]) < 7 else int(decon[4]) + 1
            elif index == 1:
                last_fy = int(decon[4]) if int(decon[3]) < 7 else int(decon[4]) + 1
        list = [x for x in range(first_fy, last_fy + 1)]
        return list
    def months_in_fy (self, fy): #need refining
        mlist = list(range(7, 13)) + list(range(1, 7))
        last_in_month, yearmonth = [], []
        for index, month in enumerate(mlist):
            count = 0 if index >= 6 else 1
            last_in_month.append(self.construct('last', fy-count, month))
            yearmonth.append(str(fy-count)+str(month).zfill(2))
        header_dates = [self.date_header(date) for date in last_in_month]
        return [header_dates, yearmonth]
    def time_diff_in_months(self, start, end):
        yeardiff = end.year - start.year
        monthdiff = end.month - start.month
        timediff = yeardiff * 12 + monthdiff + 1
        if timediff <= 0:
            timediff = 0
        return timediff
class SortData:
    def __init__(self, hd, ym):
        self.zerolist = [0]*12
        self.header_dates, self.yearmonth = hd, ym
    def title_adder(self, data, title):
        output = [[''], [title]] + data
        return output
    def monthly_income(self, items, rec):
        for row in rec:
            yield [row[item] for item in items] #reorganise data in desired output for output
    def revenue_view_ma (self, rec):
        header = [''] + self.header_dates + ['Total']
        ym = self.yearmonth
        m_total, rwm_total = self.zerolist[:], self.zerolist[:]
        for row in rec:
            for index, ym in enumerate(yearmonth):
                if row['sdym'] <= ym and row['edym'] >= ym:
                    m_total[index] += row[ym]
                    rwm_total[index] += row[ym]*row['likelihood']
        monthly_revenue = [header, ['$'] + m_total + [sum(m_total)], ['Risk Weighted$'] + rwm_total + [sum(rwm_total)]]
        monthly_revenue = self.title_adder(monthly_revenue, "Revenue View (averaged monthly income)")
        return [rwm_total, m_total, monthly_revenue]
    def deal_close_in_fy(self, rec):
        title = "Deals close(d) in this FY"
        header = [''] + self.header_dates + ['Total']
        output, res_holder = [header], {} #output is a list, res_holder is a dict for calcualtion
        items_to_cal = ['Count', '$', 'Risk Weighted$', 'Average Deal Size$']
        for item in items_to_cal:
            res_holder[item] = self.zerolist.copy()
        for row in rec:
            if row['sdym'] in self.yearmonth: #if deal closes/closed in this fiscal year
                index = self.yearmonth.index(row['sdym']) #get index of element in list
                res_holder['Count'][index] += 1
                res_holder['$'][index] += row['amt']
                res_holder['Risk Weighted$'][index] += row['rw_amt']
        for index in range (12): #Calaculate Average, i.e. 12 months in a year
            if res_holder['$'][index] != 0 and res_holder['Count'][index] != 0:
                res_holder['Average Deal Size$'][index] = res_holder['$'][index]/res_holder['Count'][index]
        for key, list in res_holder.items():
            if key == "Count":
                cnt = sum(list)
            elif key =="$":
                total = sum(list)
            if key != "Average Deal Size$":
                list.append(sum(list))
            else:
                list.append(total/cnt)
            list.insert(0, key)
            output.append(list)
        output = self.title_adder(output, title)
        return output
    def single_criteria_fy(self, rec, criteria):
        header = self.header_dates + ['FY Total$', '%']
        title = "By "+criteria.capitalize()+" (for deals closed in this FY)"
        criteria_group = list(set([row[criteria] for row in rec]))
        if criteria == 'stage':
            criteria_group = list(sorted(set(all_stages) & set(criteria_group), key = all_stages.index)) #compare the two sets (set has no order) and use index in list to sort compared result
        else:
            criteria_group.sort()
        output, res_holder = [], {} #output to store output, res_holder to store temp values during calculation
        fy_rwtotal, fy_total = 0, 0
        items_to_cal = ['$', 'Risk Weighted$']
        for group in criteria_group: #initiate dict values
            res_holder[group] = {}
            for item in items_to_cal:
                res_holder[group][item] = self.zerolist.copy()
        for row in rec: #calculation
            group = row[criteria]
            if row['sdym'] in self.yearmonth:
                index = self.yearmonth.index(row['sdym']) #get index of element in list
                res_holder[group]['$'][index] += row['amt']
                res_holder[group]['Risk Weighted$'][index] += row['rw_amt']
                fy_total += row['amt']
                fy_rwtotal += row['rw_amt']
        for item in items_to_cal: #organise format of data for output
            output.append([item] + header)
            total = self.zerolist.copy() #initate monthly total
            for group in criteria_group: #vertical total i.e. monthly
                if item == "Risk Weighted$":
                    fyt = fy_rwtotal
                elif item == "$":
                    fyt = fy_total
                output.append([group] + res_holder[group][item] + [sum(res_holder[group][item]), sum(res_holder[group][item])/fyt])
                for index, value in enumerate(res_holder[group][item]):
                    total[index] += value
            output.append(['Total $']+ total + [fyt, 1])
            output.append([''])
        output = self.title_adder(output, title)
        return [self.yearmonth, output]
    def groupby_fiscal_ad(self, rec): #Two Criteria then we are concern with all deals
        title = "By Fiscal Year (all deals)"
        dates_found = [rec['sdstr'] for rec in sale_rec]
        year_range = dh.year_range([min(dates_found), max(dates_found)])
        items_to_cal = ['Count', 'Revenue$', 'Risk Weighted$', 'Average Deal Size$'] #things to calculate
        res_holder = {}
        for item in items_to_cal:
            res_holder[item] = {}
            for year in year_range:
                res_holder[item][year] = 0
        for row in rec:
            yyyy = int(row['sdym'][0:4])
            mm = int(row['sdym'][4:])
            fy = yyyy + 1 if mm > 6 else yyyy
            res_holder['Count'][fy] += 1
            res_holder['Revenue$'][fy] += row['amt']
            res_holder['Risk Weighted$'][fy] += row['rw_amt']
        output = [[''] + year_range + ['Total']] #organise data format for output
        for year in year_range:
            res_holder['Average Deal Size$'][year] = res_holder['Revenue$'][year]/res_holder['Count'][year]
        for key, pair in res_holder.items():
            output.append([key]+ list(pair.values()) + [sum(list(pair.values()))] )
            if key == 'Count':
                t_cnt = sum(list(pair.values()))
            elif key == 'Revenue$':
                t_sum = sum(list(pair.values()))
        output[-1][-1] = t_sum/t_cnt #total average (t_sum/t_cnt) is the last element on the last line
        output = self.title_adder(output, title)
        return output
    def likelihood_ad(self, rec, lc, hc): #Two Criteria then we are concern with all deals
        title = "By Likelihood (all deals)"
        items_to_cal = ['Count', '$', 'Risk Weighted$', ] #things to calculate
        res_holder = {}
        for item in items_to_cal:
            res_holder[item] = {}
            res_holder[item]['low'] = 0
            res_holder[item]['mid'] = 0
            res_holder[item]['high'] = 0
        for row in rec: #calcaulation
            if row['likelihood'] < lc:
                group = 'low'
            elif row['likelihood'] > hc:
                group = 'high'
            else:
                group = 'mid'
            res_holder['Count'][group] += 1
            res_holder['$'][group] += row['amt']
            res_holder['Risk Weighted$'][group] += row['rw_amt']
        for key, pairs in res_holder.items():
                res_holder[key]['total'] = sum(pairs.values())
        output = [['', '<'+str(int(lc*100))+'%', str(int(lc*100))+'-'+str(int(hc*100))+'%', '>'+str(int(hc*100))+'%', 'Total']]
        for key, pair in res_holder.items():
            output.append([key]+ list(pair.values()))
        output = self.title_adder(output, title)
        return output
    def by_close_date(self, rec):
        title = "By Close Date (all deals)"
        res_holder = {}
        items_to_cal = ['Count', '$', 'Risk Weighted$', ] #things to calculate
        for item in items_to_cal:
            res_holder[item] = {}
            for ym in self.yearmonth:
                res_holder[item][yearmonth] = 0
        for row in rec: #calcaulation
            sd = row['sdstr']
            res_holder['Count'][sd] += 1
            res_holder['$'][sd] += row['amt']
            res_holder['Risk Weighted$'][sd] += row['rw_amt']
        output = [[''] + self.month_end + ['Total']] #organise data format for output
        for key, pair in res_holder.items():
            output.append([key]+ list(pair.values()) + [sum(list(pair.values()))])
        output = self.title_adder(output, title)
        return output
    def single_criteria_ad(self, rec, criteria):
        title = "By "+criteria.capitalize()+" (all deals)"
        criteria_group = list(set([row[criteria] for row in rec]))
        if criteria == 'stage':
            criteria_group = list(sorted(set(all_stages) & set(criteria_group), key = all_stages.index)) #compare the two sets (set has no order) and use index in list to sort compared result
        else:
            criteria_group.sort()
        res_holder = {}
        items_to_cal = ['Count', '$', 'Risk Weighted$'] #things to calculate
        for item in items_to_cal:
            res_holder[item] = {}
            for group in criteria_group: #initiate dict values
                res_holder[item][group] = 0
        for row in rec:
            group = row[criteria]
            res_holder['Count'][group] += 1
            res_holder['$'][group] += row['amt']
            res_holder['Risk Weighted$'][group] += row['rw_amt']
        output = [[''] + criteria_group + ['Total']] #organise data format for output
        for key, pair in res_holder.items():
            output.append([key]+ list(pair.values()) + [sum(list(pair.values()))])
        output = self.title_adder(output, title)
        return output
    def double_criteria_ad(self, rec, crit1, crit2):
        title = "By "+crit1.capitalize()+" & By "+crit2.capitalize() + " (all deals)"
        c1_group = list(set([row[crit1] for row in rec]))
        c2_group = list(set([row[crit2] for row in rec]))
        c1_group.sort()
        c2_group.sort()
        if crit1 == 'stage': #custom order is required if criteria == stage
            c1_group = list(sorted(set(all_stages) & set(c1_group), key = all_stages.index)) #compare the two sets (set has no order) and use index in list to sort compared result
        elif crit2 == 'stage':
            c2_group = list(sorted(set(all_stages) & set(c2_group), key = all_stages.index)) #compare the two sets (set has no order) and use index in list to sort compared result
        res_holder = {}
        items_to_cal = ['Count', '$', 'Risk Weighted$'] #things to calculate
        for c1 in c1_group: #initiate dict values
            res_holder[c1] = {}
            for c2 in c2_group:
                res_holder[c1][c2] = {}
                for item in items_to_cal:
                    res_holder[c1][c2][item] = 0
        for row in rec:
            c1 = row[crit1]
            c2 = row[crit2]
            res_holder[c1][c2]['Count'] += 1
            res_holder[c1][c2]['$'] += row['amt']
            res_holder[c1][c2]['Risk Weighted$'] += row['rw_amt']
        output = [['']+items_to_cal] #organise data format for output
        for c1, pairs in res_holder.items():
            group = [c1] + [0] * len(items_to_cal) #list [Prosepct, 0, 0, 0]
            subgroup = []
            for c2, values in pairs.items():
                subgroup.append([c2] + list(values.values())) #list [ANZ, cnt, $, $RWS]
            for row in subgroup:
                for x in range(1, len(row)):
                    group[x] += row[x]
            output.append(group)
            for row in subgroup:
                output.append(row)
            output.append([''])
        return output
fy = int(input("\nEnter the Fiscal Year in concern: \n>>"))
filetype = ".csv"
fn =  input("\nType in the name of your source file (no file extension needed): \n>>")
filename = fn + filetype
lc = float(input("\n\tPlease specific a lower limit for likelihood summary i.e. type in 0.5 for 50% or less\n\t>>"))
hc = float(input("\n\tPlease specific a upper limit for likelihood summary i.e. type in 0.75 for 75% or more\n\t>>"))
#---Open csv source file
with open(filename, encoding='ISO-8859-1') as csvfile: #Use encoding ISO-8859-1 to avoid problems caused by accented characters i.e. French/Spanish brands
    readCSV = csv.reader(csvfile, delimiter=',')
    headers = next(readCSV)
    head = headers[:]
    head.insert(3, "Region") #insert at position 3 and shift original elements right
    sale_rec = [] #---Initiate variables/lists for storing data from source file
#---Read input begins
    dh = DateHandler()
    if fy != 0:
        month_end, yearmonth = dh.months_in_fy(fy)
    for row in readCSV:
        line = {}
        subject = ['id', 'deal', 'person', 'stage', 'company', 'product', 'quote', 'potential']
        for x in range(len(subject)):
            line[subject[x]] = row[x]
        line['person'] = row[2].title() #sort out random casing of names i.e. some names are all cap or not capitalised
        line['region'] = region_dict[row[2].title()]
        vol = float(0) if not row[8] else float(row[8]) #To avoid halt of operation due to division performed on empty value
        line['amt'] = vol
        #---Format likelihood - input as float. get rid of % sign and set empty field with dummy figure to avoid operating on empty value
        like = str(38.88) if not row[9] else row[9] #set dummy percentage as 38.88% if none is provided in the source
        non_decimal = re.compile(r'[^\d.]+') #regex to get rid of any non numerical values. decimal place is allowed
        #Alternative solution: filter( lambda x: x in '0123456789.', s ) but regex is faster
        likeli = non_decimal.sub('', like) #syntax: re.sub(pattern,repl,string)| Replace non decimal char with empty string
        line['likelihood'] = float(likeli)/100
        line['rw_amt']= vol * float(likeli)/100
        sd = dh.deconstruct(dh.findformat(row[10]))
        ed = dh.deconstruct(dh.findformat('2000-1-1')) if not row[11] else dh.deconstruct(dh.findformat(row[11]))
        line['sdstr'] = sd[1] # 0 date, 1 str, 2 day, 3 month, 4 year
        line['edstr'] = ed[1]
        line['sdym'] = sd[4] + sd[3]
        line['edym'] = ed[4] + ed[3]
        timediff = dh.time_diff_in_months(sd[0], ed[0])
        if vol != 0 and timediff != 0:
            spc = vol / timediff #sale income per delivery cycle
        else:
            spc = 0 #0/0 is infinitive, so override by setting result to zero to avoid error
        for ym in yearmonth:
            if line['sdym'] <= ym and line['edym'] >= ym:
                line[ym] = spc
            else:
                line[ym] = 0
        sale_rec.append(line)
print(f"\n{star[0:6]}{space[0:4]}{star[0:6]}{space[0:4]}{star[0:6]}{space[0:4]}OK, found your file.{space[0:4]}{star[0:6]}{space[0:4]}{star[0:6]}{space[0:4]}{star[0:6]}")
#---End of reading & processing of source file
start = datetime.now()#----Start timer
print (f"\n{arrow[0:7]}Calculation Begins{arrow[0:55]}")
process = SortData(month_end, yearmonth)
chart = PlotGraph()
chart_label = [str(me[3:6]) + " " + str(me[7:]) for me in month_end]
output_list = ['id', 'deal', 'person', 'region', 'stage', 'company', 'product', 'quote', 'potential', 'amt', 'likelihood', 'sdstr', 'edstr'] + yearmonth
monthly_income = [head + month_end]
monthly_income.extend(list(process.monthly_income(output_list, sale_rec)))
rwm_total, m_total, by_revenue_view = process.revenue_view_ma(sale_rec)
title = 'Revenue View VS Budget'
figname = 'revenue_view.png'
chart.bar_and_line(chart_label, [budget], [rwm_total, m_total], ['Risk Weighted US$', 'Revenue US$', 'Budget US$'], title, figname)

by_fiscal_ad = process.groupby_fiscal_ad(sale_rec)
header = by_fiscal_ad[2][1:-1] #list of all fiscal years with deals
bar_list = [by_fiscal_ad[4][1:-1], by_fiscal_ad[5][1:-1]] #RW$ & $ without row header and subtotal
ind = [by_fiscal_ad[4][0], by_fiscal_ad[5][0]] #labels of row i.e. $RW$ and $
title = 'Revenue View by Fiscal Year'
figname = 'fiscalyear_view.png'
chart.bar_chart(header, bar_list, ind, title, figname)#linebreak, title, header, count, rw_sum, sum, avg

by_Likelihood_ad = process.likelihood_ad(sale_rec, lc, hc)

by_stage_ad = process.single_criteria_ad(sale_rec, 'stage')

by_deal_close_fy = process.deal_close_in_fy(sale_rec)

by_stage_fy = process.single_criteria_fy(sale_rec, 'stage')[1]

by_sale_person_fy = process.single_criteria_fy(sale_rec, 'person')[1]

graph_header, by_region_fy = process.single_criteria_fy(sale_rec, 'region')
#col = by_region_fy[2:3][0][:-2].copy()
df_rw= pd.DataFrame(data = [by_region_fy[x][:-2] for x in range(3,9)], columns = [''] + chart_label) #move the dataframe definition outside of the function because the resouce data contains both normal and rw results
df_rw= df_rw.set_index(df_rw.columns[0])
df_rw.index.name = None #to remove the sub-index
title_rw = 'FY '+ str(fy) +' revenue by deal close month & region (Risk-Weighted)'
fn_rw = 'region_risk_weight_stacked_bar.png'
chart.stacked_bar(df_rw.T, title_rw, 'region', '#EEE4DB', '#FFFFFF', fn_rw)
df_s = pd.DataFrame(data = [by_region_fy[x][:-2] for x in range(12,18)], columns = [''] + chart_label)
df_s = df_s.set_index(df_s.columns[0])
title_s = 'FY '+ str(fy) +' revenue by deal close month & region (Non-Weighted)'
fn_s = 'region_risk_non_weight_stacked_bar.png'
chart.stacked_bar(df_s.T, title_s, 'region', '#FFFFFF', '#D8D8D8', fn_s)

by_stage_n_region = process.double_criteria_ad(sale_rec, 'stage', 'region')
chart.pie_chart(by_stage_n_region[0:-1], 8, 'region', 'stage_n_region.png')
chart.donut_chart(by_stage_n_region[0:-1], 8, 'stage_n_region_donut.png')
#---Done running functions
print (f"\n\t\t\tRows Processed: \n\t\tProcessing Time: {datetime.now()-start}")
print (f"\n{arrow[0:51]}Calculation Completed{arrow[0:8]}")
outputting=input("\nPlease enter a new filename if you would like to save results: \n>>")
#---Write result to files
with open((outputting+filetype), 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(monthly_income)
    writer.writerows(by_fiscal_ad)
    writer.writerows(by_Likelihood_ad)
    writer.writerows(by_stage_ad)
    writer.writerows(by_revenue_view)
    writer.writerows(by_deal_close_fy)
    writer.writerows(by_stage_fy)
    writer.writerows(by_region_fy)
    writer.writerows(by_stage_n_region)
#---End of writing to output file
print(f"\n{hash}")
if not outputting:
    print(f"\n{space[0:32]}See you next time!\n")
else:
    print (f"\n{space[0:6]}Your output file can be found in {Path(__file__).parent.absolute()}/{outputting+filetype}\n")
print (f"{hash[0:35]}All Done!{hash[0:36]}\n")
