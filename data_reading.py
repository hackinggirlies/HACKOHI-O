import pandas as pd
df = pd.read_csv("CORE_HackOhio_subset_cleaned_downsampled 1.csv")
print(df.head()) 
#DATETIME_DTM	Datetime, date and time when observation was recorded
#PNT_NM	Point name, the safety criteria being assessed. Forms a primary key when combined with OBSRVN_NB (observation number)
#QUALIFIER_TXT	Qualifier text, list of predetermined observations chosen by the reviewer based on the point being assessed
#PNT_ATRISKNOTES_TX	Point at-risk notes text, comments left by observer regarding unsafe conditions they found
#PNT_ATRISKFOLWUPNTS_TX	Point at-risk follow up notes text, recommended remediation for at-risk conditions observed

date_time = df['DATETIME_DTM']
#Date time format looks like this: 3/15/2023 11:01
safety_criteria = df['PNT_NM']
#format looks like this: Equipment - Tools &Equipment
#a few of these are formatted as questions like this: 
#Did you recognize additional sprain or strain hazards that had not already been recognized and mitigated? If so, please select or describe these hazards in the At-Risk notes. 1
qualifier = df['QUALIFIER_TXT']
#examples: Assistance requested, Personal voltage detector
comments = df['PNT_ATRISKNOTES_TX']
followups = df['PNT_ATRISKFOLWUPNTS_TX']