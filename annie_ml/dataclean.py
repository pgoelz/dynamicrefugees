import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


def edu_groups_numeric(series):
    if series in ['Unknown', 'NONE', '', np.NaN]:
        return 0
    elif series in ['Kindergarten', 'Primary', 'Intermediate']:
        return 1
    elif series in ['Secondary']:
        return 2
    elif series in ['Pre-University', 'Technical School', 'Professional']:
        return 3
    elif series in ['Graduate School', 'University/College']:
        return 4
    else:
        raise ValueError("Unknown education group: " + repr(series))


def edu_group_names(number):
    dictionary = {0: '0-Unknown/None', 1: '1-Less than Secondary', 2: '2-Secondary', 3: '3-Advanced', 4: '4-University'}
    return dictionary[number]


def language_profc(series):
    if series in ['None']:
        return 1
    elif series in ['Some']:
        return 2
    elif series in ['Good']:
        return 3
    else:
        return 0


def parse_date(string):
    return pd.Timestamp(datetime.strptime(string, "%m/%d/%Y"))


def fiscal_year(date: pd.Timestamp):
    if date.month >= 10:
        return date.year + 1
    else:
        return date.year


def clean_affiliates(affiliate_name: str) -> str:
    if affiliate_name.strip() != affiliate_name:
        affiliate_name = affiliate_name.strip()
    if '-' not in affiliate_name:
        print('Weird affiliate name: ' + repr(affiliate_name))
    if affiliate_name.startswith("X "):
        affiliate_name = affiliate_name[2:]
    if affiliate_name.endswith("_X"):
        affiliate_name = affiliate_name[:-2]
    if affiliate_name == "OH-Cleveland":
        return "OH-Cleveland Heights"
    return affiliate_name


def calculate_age(born, date=None):
    if date is None:
        dateage = pd.Timestamp(datetime.today())
        return (dateage - born).astype('timedelta64[Y]')
    else:
        dateage = pd.to_datetime(date)
        born = pd.to_datetime(born)
        return (dateage - born).astype('timedelta64[Y]')


_data_cache = None
def get_df(rawfile: str):
    global _data_cache

    if _data_cache is None:
        data = pd.read_csv(rawfile)
    else:
        data = _data_cache

    data = data.dropna(subset=['Case Number', 'Seq', 'Case Pool', 'Arrival Date', 'Allocation Date', 'DOB'])
    data.columns = [x.lower() for x in data.columns]

    data['allocation date'] = data['allocation date'].apply(parse_date)
    data['arrival date'] = data['arrival date'].apply(parse_date)
    data['arrival date'] = data[['allocation date', 'arrival date']].max(axis=1)
    data['fy'] = data['arrival date'].apply(fiscal_year)
    data['affiliate'] = data['affiliate'].apply(clean_affiliates)

    return data


def historic_data_to_agents(rawfile: str, min_allocation_date: Optional[datetime.date] = None,
                            max_allocation_date: Optional[datetime.date] = None,
                            min_arrival_date: Optional[datetime.date] = None,
                            max_arrival_date: Optional[datetime.date] = None, only_free: bool = False) -> pd.DataFrame:
    """Convert cases in the format of `Orgdata_seri2_with_CN.csv` ("historic format") into the agent format.

    Historic data has the following columns:
      Seq,Case Pool,Relat Code,Gender Code,Nationality Code,English Speaking,DOB,Education Level,Arrival Date,Affiliate,
      Medical Condition,Treatment Urgency,Employed,Urgency Code,Language,Native Language,Proficiency Level Read,
      Proficiency Level Speak,Proficiency Level Write,Case Number

    We return columns:
      'case number'
      'seq'
      'relat code' (string): e.g., "PA", "HU", "WI", "SO", "DA", ... Relat code is not in the arrival format.
      'age' (int)
      'male' (int): 0 (female) or 1 (male)
      'continent' (string): continent \in ['africa', 'middle east', 'other continent']
      'english proficiency' (int): 0-3, higher is better
      'education level' (string): education level \in ['0-Unknown/None', '1-Less than Secondary', '2-Secondary',
                                                       '3-Advanced', '4-University']
      'number conditions' (int): clipped at 5, so 0-5
      'number languages' (int): clipped at 3, so 0-3
      'urgency code' (int): 0-1, all codes except for 'NOR' map to a level of 1
      'languages spoken well' (set of strings): all languages spoken better than level 2 by the individual
      'nationality code' (str)
      'fy' (int)
      'historic affiliate' (str)
      'arrival date' (pd.Timestamp)
      'allocation date' (pd.Timestamp)

    Additional columns depending on arguments:
      'ties to' (string): unless `only_free`. For US tie cases, their affiliate. For free cases, the empty string.

    Intentionally not contained are employed, dob, case pool, affiliate, arrival date (unless `keep_arrival`).

    The following features appear in the format of arrivals, but are not part of our output since they can be inferred
    from present information:
      'pa' (int): 0-1; relat code == 'PA'?
      'medical condition' (int): 0-1; number conditions.clip(0, 1)
      'english' (int): 0-1; english proficiency >= 2?

    The following features appear in the format of arrivals, but are inferred from cases rather than individuals:
      'case size' (int): i think unclipped
      'n kids' (int): 0-5 fed into ML, but more might be needed for generation
      'single parent' (int): 0-1
      'couple' (int): 0-1

    These features will be generated in `agents_to_features()`.

    Args:
        rawfile (str): Path to input file, e.g., `Orgdata_seri2_with_CN.csv`
        min_allocation_date: if present, ignore all cases allocated strictly before this date
        max_allocation_date: if present, ignore all cases allocated from this date on
        min_arrival_date: if present, ignore all cases arriving strictly before this date
        max_arrival_date: if present, ignore all cases arriving from this date on
        only_free: ignore all cases that are not in the free case pool

    Returns:
        pandas.DataFrame with indicated columns.
        :param min_arrival_date:
        :param max_arrival_date:
    """
    data = get_df(rawfile)
    if min_allocation_date is not None:
        data = data[data['allocation date'] >= pd.Timestamp(min_allocation_date)]
    if max_allocation_date is not None:
        data = data[data['allocation date'] < pd.Timestamp(max_allocation_date)]
    if min_arrival_date is not None:
        data = data[data['arrival date'] >= pd.Timestamp(min_arrival_date)]
    if max_arrival_date is not None:
        data = data[data['arrival date'] < pd.Timestamp(max_arrival_date)]
    if only_free:
        data = data[data['case pool'] == 'Free']

    data['education level'] = data['education level'].apply(edu_groups_numeric)
    data = data.drop_duplicates(['case number', 'seq', 'medical condition', 'language', 'education level'])

    data['proficiency'] = data['proficiency level speak'].apply(language_profc)
    data['good language'] = np.where(data['proficiency'] >= 2, data['language'], np.nan)
    languages = pd.read_csv("annie_ml/language_codes.csv", index_col="Language Name")
    languages.index = languages.index.str.lower().str.replace("[^a-z]", "")
    language_codes = languages["Language Code"].to_dict()
    data['good language code'] = data['good language'].str.lower().str.replace("[^a-z]", "").map(lambda x:
                                                                                        language_codes.get(x, x))
    data['english proficiency'] = np.where(data['language'] == 'English', data['proficiency'], 0)

    grouped = data.groupby(by=['case number', 'seq'])

    data = data.drop_duplicates(['case number', 'seq'])

    data['education level'] = grouped['education level'].transform(np.max).apply(edu_group_names)  # apply needed to later dummify
    data['number conditions'] = grouped['medical condition'].transform("nunique").clip(0, 3)
    data['number languages'] = grouped['good language'].transform("nunique").clip(0, 3)
    languages = grouped['good language code'].agg(lambda x: set(x.dropna())).rename("languages spoken well")
    data = data.join(languages, on=["case number", "seq"])
    data['english proficiency'] = grouped['english proficiency'].transform(np.max)

    data['age'] = calculate_age(data['dob'], data['allocation date'])
    data['male'] = data['gender code'].replace(['F', 'M'], [0, 1])
    data['urgency code'] = np.where(data['urgency code'] == 'NOR', 0, 1)

    # Geographical origin
    current_dir = os.path.dirname(__file__)
    key = pd.read_csv(os.path.join(current_dir, 'nat_codes.csv'))[['nationality code', 'continent']]

    data['nationality code'] = data['nationality code'].str.lower()
    data = data.merge(key, how='left', left_on='nationality code', right_on='nationality code')
    data['continent'] = np.where(data['continent'].isin(['africa', 'middle east']), data['continent'],
                                 'other continent')
    data['ties to'] = np.where(data['case pool'].isin(['Geo', 'Predestined', 'SIV U.S. Ties']), data['affiliate'], '')

    data['historic affiliate'] = data['affiliate']

    columns = ['case number', 'seq', 'relat code', 'age', 'male', 'continent', 'english proficiency',
               'education level', 'number conditions', 'number languages', 'urgency code', 'languages spoken well',
               'nationality code', 'fy', 'historic affiliate', "arrival date", "allocation date"]
    if not only_free:
        columns.append('ties to')
    return data[columns]


def agents_to_features(agent_table, synergy_list):
    # expects features:
    # • all from historic_data_to_agents():
    #       'case number', 'seq', 'relat code', 'age', 'male', 'continent', 'english proficiency', 'education level',
    #       'number conditions', 'number languages', 'urgency code', 'arrival date', "allocation date"
    # • 'affiliate'
    # all family members need to be fed in to calculate family features
    # same case can be evaluated for multiple affiliates, but not multiple allocation dates in single call

    data = agent_table.copy()
    data = data.drop(columns=['languages spoken well', 'nationality code', 'fy', 'historic affiliate'])
    data['pa'] = np.where(data['relat code'] == 'PA', 1, 0)

    data['is_parent'] = np.where(data['relat code'].isin(['PA', 'HU', 'WI']), 1, 0)
    data['is_child'] = np.where((~data['is_parent'] & (data["age"] < 18)), 1, 0)

    grouped = data.groupby(by=['case number', 'affiliate'])

    data['case size'] = grouped['seq'].transform('count').clip(0, 6)
    data['n kids'] = grouped['is_child'].transform('sum').clip(0, 5)
    data['n parents'] = grouped['is_parent'].transform('sum')
    data['single parent'] = np.where((data['n parents'] == 1) & (data['n kids'] > 0), 1, 0)
    data['couple'] = np.where(data['n parents'] > 1, 1, 0)

    data['medical condition'] = data['number conditions'].clip(0, 1)
    data['english'] = np.where(data['english proficiency'] >= 2, 1, 0)

    data['affiliate to dummify'] = data['affiliate']
    dummify_features = {'affiliate to dummify': synergy_list,
                        'continent': ['africa', 'middle east'],
                        'education level': ['0-Unknown/None', '2-Secondary', '3-Advanced', '4-University']}  # 1=default
    dummies = {}
    for feature, values in dummify_features.items():
        data[feature] = np.where(data[feature].isin(values), data[feature], np.nan)
        data[feature].astype(pd.api.types.CategoricalDtype(categories=values))
        dummies[feature] = pd.get_dummies(data[feature])
        # make sure all values appear as columns even if not present
        dummies[feature] = dummies[feature].reindex(columns=values,
                                                    fill_value=0)
        data = data.merge(dummies[feature], left_index=True, right_index=True)
    affiliates = dummies["affiliate to dummify"]

    # Polynomials
    data['age2'] = data['age'] ** 2

    # Create interactions across features
    # i.edu#(i.english i.male)
    data = add_interactions(data, 'education level', 'english speaking')
    data = add_interactions(data, 'education level', 'male')
    # (c.age c.nkids)#i.male
    data = add_interactions(data, 'n kids', 'male', varname1_cont=True)
    data = add_interactions(data, 'age', 'male', varname1_cont=True)
    # i.pa#i.male
    data = add_interactions(data, 'pa', 'male')
    # i.single parent#i.male
    data = add_interactions(data, 'single parent', 'male')
    # c.nconditions#i.male
    data = add_interactions(data, 'number conditions', 'male', varname1_cont=True)
    # add a constant term
    data['constant'] = 1

    data['arrival date'] = pd.to_datetime(data['arrival date']).dt.to_period('M').map(str)
    data['allocation date'] = pd.to_datetime(data['allocation date']).dt.to_period('M').map(str)

    # Import macrodata and add macro-characteristics interactions
    macrodata = return_macrodata()
    macro_vars = macrodata.keys().tolist()
    macro_vars.remove('allocation date')
    macro_vars.remove('affiliate')

    macrodata = macrodata[macro_vars + ['affiliate', 'allocation date']]
    data = data.reset_index().merge(macrodata, on=['affiliate', 'allocation date']).set_index('index')

    numeric_data = data.drop(errors='ignore',
        columns=macro_vars + ['case number', 'seq', 'relat code', 'affiliate to dummify', 'continent', 'affiliate',
                              'arrival date', 'allocation date', 'education level', '1.education level#1.male',
                              'ties to'])

    for var in macro_vars:
        temp = numeric_data.mul(data[var], axis='index').add_suffix('#{}'.format(var))
        data = data.merge(temp, left_index=True, right_index=True)

    # Add affiliate interactions
    for aff in list(affiliates.keys()):
        temp = numeric_data.multiply(affiliates[aff], axis="index").add_suffix('#{}'.format(aff))
        data = data.merge(temp, left_index=True, right_index=True)

    data = data.sort_index()
    return data


def return_macrodata():
    # Load affiliate/county info
    affiliate_county = get_affiliate_counties()
    counties = affiliate_county['county']
    county_names = counties.values.tolist()
    periods = pd.date_range('2011-01-01', datetime.today(), freq='MS').to_period('M')
    end_year = periods[-3].year
    start_year = 2011

    # Load macrodata
    current_dir = os.path.dirname(__file__)
    if os.path.isfile(os.path.join(current_dir, 'data/macrodata.csv')) is False:
        macro = update_macrodata(start_year, end_year, counties, periods)
    else:
        macro = pd.read_csv(os.path.join(current_dir, 'data/macrodata.csv'))

    update = False
    # Check if time range ok
    today = str(pd.to_datetime(datetime.today()).to_period('M'))
    counties_in_macro = macro['county'].drop_duplicates().values.tolist()

    if today not in macro['time'].values:
        update = True
    if set(county_names) != set(counties_in_macro):
        update = True

    # Since it seems that official unemployment statistics change many years afterwards, deactivate updates for the sake
    # of reproducibility.
    if False:  # update:
        macro = update_macrodata(start_year, end_year, counties, periods)

    # Recode with affiliates
    macro = macro.merge(affiliate_county, how='left', on=['county']).drop(columns=['county'])
    macro = macro.rename(columns={'time': 'allocation date'})
    macro = macro.drop_duplicates()
    return macro


def update_macrodata(start_year, end_year, counties, periods):
    print('Updating macro data: This step might take a few minutes')

    raw = get_unemployment_data(start_year, end_year)
    raw['time'] = pd.to_datetime(raw[['year', 'month']].assign(day=1)).dt.to_period('M')
    raw = raw[['time', 'county', 'unemployment rate']]

    index = pd.MultiIndex.from_product([periods, counties], names=["time", "county"])
    skel = pd.DataFrame(index=index).sort_values(by=['county', 'time']).reset_index()

    raw = pd.merge(skel, raw, how='left')

    raw['unemployment rate'] = raw.groupby('county')['unemployment rate'].shift(2)
    raw['time'] = raw['time'].map(str)

    current_dir = os.path.dirname(__file__)
    raw.to_csv(os.path.join(current_dir, 'data/macrodata.csv'), index=False)
    return raw


def get_unemployment_data(start_year, end_year):
    df_areas = pd.read_csv('https://download.bls.gov/pub/time.series/la/la.area', sep='\t')

    # Only keep county information
    df_areas = df_areas.loc[df_areas['area_type_code'].str.contains('F')]
    df_areas.reset_index(drop=True, inplace=True)

    # Get county and state information
    df_areas['county'] = df_areas['area_text']

    # Remove whitespace
    df_areas['area_code'] = df_areas['area_code'].map(lambda x: x.strip())
    df_areas['county'] = df_areas['county'].map(lambda x: x.strip())

    # Remove unnecessary columns
    df_areas = df_areas[['area_code', 'county']]

    df_unemp_10_14 = get_BLS_county_data('https://download.bls.gov/pub/time.series/la/la.data.0.CurrentU10-14',
                                         df_areas)
    df_unemp_15_19 = get_BLS_county_data('https://download.bls.gov/pub/time.series/la/la.data.0.CurrentU15-19',
                                         df_areas)
    df_unemp_20_24 = get_BLS_county_data('https://download.bls.gov/pub/time.series/la/la.data.0.CurrentU20-24',
                                         df_areas)

    df_unemp_county = df_unemp_10_14
    df_unemp_county = df_unemp_county.append(df_unemp_15_19)
    df_unemp_county = df_unemp_county.append(df_unemp_20_24)

    df_unemp_county = df_unemp_county.sort_values(by=['area_code', 'year', 'month'], axis=0)
    df_unemp_county = df_unemp_county[
        (df_unemp_county['year'] >= int(start_year)) & (df_unemp_county['year'] <= int(end_year))]
    df_unemp_county = df_unemp_county[['area_code', 'county', 'year', 'month', 'unemployment rate']]

    return df_unemp_county


def get_BLS_county_data(BLS_data_path, df_areas):
    """
    BLS_data_path : path for the text file containing the BLS data
    df_areas      : dataframe containing BLS information about counties/areas
    """
    # Import area information
    headers = ['series_id', 'year', 'period', 'value', 'footnote_codes']
    # col_types = {'series_id': str, 'year': int, 'period': str, 'value': str, 'footnote_codes': str}
    # col_types = [str, int, str, str, str]
    df_bls_county = pd.read_csv(BLS_data_path, skiprows=1, names=headers, sep='\t')

    # Remove white space from code..
    df_bls_county['series_id'] = df_bls_county['series_id'].map(lambda x: x.strip())

    # Convert 'value' to numeric (kind of slow...)
    df_bls_county['value'] = df_bls_county['value'].apply(pd.to_numeric, errors='coerce')

    # Get variable code
    df_bls_county['var_code'] = df_bls_county['series_id'].str[-2:]

    # Get area code
    df_bls_county['series_id'] = df_bls_county['series_id'].astype(str).str[3:].str[:-2]

    # Get FIPS code (as string to preserve initial zeros)
    df_bls_county['FIPS'] = df_bls_county['series_id'].str[2:7]

    # ------------------------------------------------------------
    # Only keep rows corresponding to counties
    df_bls_county = df_bls_county.loc[df_bls_county['series_id'].str.contains('CN')]

    # Drop columns, reset index
    df_bls_county = df_bls_county[['series_id', 'year', 'period', 'value', 'var_code', 'FIPS']]
    df_bls_county.reset_index(drop=True, inplace=True)

    # Rename codes with variable names, rename columns
    df_bls_county['var_code'] = df_bls_county['var_code'].map({'03': 'unemployment rate', '04': 'Unemployment',
                                                               '05': 'Employment', '06': 'Labor_Force'})
    df_bls_county.columns = ['area_code', 'year', 'month', 'value', 'variable_name', 'FIPS']
    df_bls_county = df_bls_county.loc[df_bls_county['month'] != 'M13']

    # Convert month to numeric values
    df_bls_county['month'] = pd.to_numeric(df_bls_county['month'].str[1:])

    # ------------------------------------------------------------
    # Merge area names and data
    df_bls_county = pd.merge(df_bls_county, df_areas, how='inner', on='area_code')

    # Convert to wide-format table
    df_bls_county = df_bls_county.pivot_table(values='value', index=['area_code', 'FIPS', 'county',
                                                                     'year', 'month'], columns='variable_name')
    df_bls_county.reset_index(inplace=True)
    df_bls_county.columns.name = None

    # ------------------------------------------------------------

    return df_bls_county


def get_affiliate_counties():
    current_dir = os.path.dirname(__file__)
    membership_df = pd.read_csv(os.path.join(current_dir, 'data/affiliates_counties.csv'))
    return membership_df


def add_interactions(df, varname1, varname2, varname1_cont=False, varname2_cont=False):
    if not varname1_cont:
        vars1 = [col for col in df if col.startswith(varname1)]
    else:
        vars1 = [varname1]
    if not varname2_cont:
        vars2 = [col for col in df if col.startswith(varname2)]
    else:
        vars2 = [varname2]
    interactions = pd.DataFrame(index=df.index)
    for counter1, var1 in enumerate(vars1):
        if varname1_cont:
            pref1 = 'c.'
        else:
            pref1 = '{}.'.format(counter1 + 1)
        for counter2, var2 in enumerate(vars2):
            if varname2_cont:
                pref2 = 'c.'
            else:
                pref2 = '{}.'.format(counter2 + 1)
            interactions['{0}{1}#{2}{3}'.format(pref1, varname1, pref2, varname2)] = df[var1] * df[var2]

    newdf = df.merge(interactions, left_index=True, right_index=True)
    return newdf
