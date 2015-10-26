import numpy as np
import pandas as pd
import math


def inv_log(x):
    return np.log(1 / (1 + x))

def spd_cap(hits,bin,size):
    if hits<bin*size:
        return 1
    else:
        return 0

def add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    #for i in xrange(1,20):
    #    df['spd_cap_%d'%i]=df.apply(lambda row: spd_cap(row['SPDhits'],i,5), axis=1)

    df['distance_sec_vtx2']=df.apply(lambda row: math.log(row['IP']/(1.0001-row['dira'])), axis=1)

    df['min_DCA']=df.apply(lambda row: min([row['DOCAone'], row['DOCAtwo'], row['DOCAthree']]), axis=1)
    df['max_DCA']=df.apply(lambda row: max([row['DOCAone'], row['DOCAtwo'], row['DOCAthree']]), axis=1)

    df['max_track_chi2']=df.apply(lambda row: max([row['p0_track_Chi2Dof'], row['p1_track_Chi2Dof'], row['p2_track_Chi2Dof']]), axis=1)

    #isolationa to isolationf is pair-wised isolation variable M_iso
    df['min_isolation_a_f']=df.apply(lambda row: min([row['isolationa'], row['isolationb'], row['isolationc'],row['isolationd'], row['isolatione'], row['isolationf']]), axis=1)
    df['max_isolation_a_f']=df.apply(lambda row: max([row['isolationa'], row['isolationb'], row['isolationc'],row['isolationd'], row['isolatione'], row['isolationf']]), axis=1)

    df['min_isobdt']=df.apply(lambda row: min([row['p0_IsoBDT'], row['p1_IsoBDT'], row['p2_IsoBDT']]), axis=1)
    df['max_isobdt']=df.apply(lambda row: max([row['p0_IsoBDT'], row['p1_IsoBDT'], row['p2_IsoBDT']]), axis=1)

    df['min_CDF']=df.apply(lambda row: min([row['CDF1'], row['CDF2'], row['CDF3']]), axis=1)
    df['max_CDF']=df.apply(lambda row: max([row['CDF1'], row['CDF2'], row['CDF3']]), axis=1)
    print 'added min max'
    
    #in Ds preselection part
    df['sum_sqrt_track_chi2']=df.apply(lambda row: math.sqrt(row['p0_track_Chi2Dof'])+math.sqrt(row['p1_track_Chi2Dof'])+math.sqrt(row['p2_track_Chi2Dof']), axis=1)

    print 'added sums'

    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3

    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    print 'added IP ratio'

    return df

filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',
              'SPDhits',
              'p0_track_Chi2Dof',
              'CDF1', 'CDF2', 'CDF3',
              'isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 
              'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',
              'DOCAone', 'DOCAtwo', 'DOCAthree']
