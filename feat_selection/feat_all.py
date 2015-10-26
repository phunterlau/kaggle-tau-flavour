#just add all features to each file, since some features may need much time
import numpy as np
import pandas as pd
import math


def inv_log(x):
    if x>0:
        return np.log(1 / (1 + x))
    else:
        return 0

#2*pt1*pt2*(cosh(eta1-eta2)-cos(phi1-phi2)) = transverse invariant mass
def inv_mass(pt1,pt2,eta1,eta2,sign): #assume back to back (phi1-phi2=pi)
    return math.sqrt(2*pt1*pt2*(math.cosh(eta1-eta2)-1*sign))

def spd_cap(hits,bin,size):
    if hits<bin*size:
        return 1
    else:
        return 0

#pseudo energy conservation on tau vertex m_tau^2+p_tau^2=sigma(m_mui^2+p_mui^2)
def pseudo_three_body_inv_mass(p0,p1,p2,pt_tau,dira):
    muon_mass = 105.6
    p_tau = pt_tau/(1.0000000001-dira)
    ret = 3*muon_mass**2+p0**2+p1**2+p2**2-p_tau**2
    if ret<0:
        return inv_log(math.sqrt(-ret))
    else:
        return 0

def pseudo_trans_three_body_inv_mass(p0,p1,p2,pt_tau):
    muon_mass = 105.6
    p_tau = pt_tau
    ret = 3*muon_mass**2+p0**2+p1**2+p2**2-p_tau**2
    if ret<0:
        return inv_log(math.sqrt(-ret))
    else:
        return 0

def get_pz(p,pt):
    return p**2-pt*2

def three_body_inv_mass(p0,p1,p2,pt_tau,pz_tau):
    muon_mass = 105.6
    return 3*muon_mass**2+p0**2+p1**2+p2**2-pt_tau**2-pz_tau**2

def add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['log_lifetime']=df.apply(lambda row: inv_log(row['LifeTime']), axis=1)

    #golden feats!
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']

    #for i in xrange(1,20):
    #    df['spd_cap_%d'%i]=df.apply(lambda row: spd_cap(row['SPDhits'],i,5), axis=1)

    #new features
    df['3body_inv_mass']=df.apply(lambda row: (pseudo_three_body_inv_mass(row['p0_p'],row['p1_p'],row['p2_p'],row['pt'],row['dira'])), axis=1)
    df['3body_trans_inv_mass']=df.apply(lambda row: (pseudo_trans_three_body_inv_mass(row['p0_pt'],row['p1_pt'],row['p2_pt'],row['pt'])), axis=1)

    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['sum_dimuon_ip']=df.apply(lambda row: row['IP_p0p2'] + row['IP_p1p2'], axis=1)
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3.
    df['NEW_IP_dira']=df['IP']*df['dira']

    df['p1p2_eta']=df.apply(lambda row: (row['p1_eta']-row['p2_eta']), axis=1)
    df['p0p1_eta']=df.apply(lambda row: (row['p0_eta']-row['p1_eta']), axis=1)
    df['p2p0_eta']=df.apply(lambda row: (row['p2_eta']-row['p0_eta']), axis=1)

    df['pseudo_invmass12']=df.apply(lambda row: inv_mass(row['p1_pt'],row['p2_pt'],row['p1_eta'],row['p2_eta'],-1), axis=1)
    df['pseudo_invmass02']=df.apply(lambda row: inv_mass(row['p0_pt'],row['p2_pt'],row['p0_eta'],row['p2_eta'],-1), axis=1)
    #the pair muon from Z' kinematic makes sense, while pairing with the 3rd doesn't have strong signal
    #although Z' doesn't have a spike of invariant mass because it is virtual, it still has some separation
    df['pseudo_invmass01']=df.apply(lambda row: inv_mass(row['p0_pt'],row['p1_pt'],row['p0_eta'],row['p1_eta'],1), axis=1)

    print 'added eta diff'

    df['pt0_tau_diff']=df.apply(lambda row: row['p0_pt']-row['pt'], axis=1)
    df['pt1_tau_diff']=df.apply(lambda row: row['p1_pt']-row['pt'], axis=1)
    df['pt2_tau_diff']=df.apply(lambda row: row['p2_pt']-row['pt'], axis=1)

    #golden features
    df['distance_sec_vtx0']=df.apply(lambda row: math.log(row['IP']/(1.0000000001-row['dira'])), axis=1)
    df['distance_sec_vtx1']=df.apply(lambda row: math.log(row['IP']/(1.0000001-row['dira'])), axis=1)
    df['distance_sec_vtx2']=df.apply(lambda row: math.log(row['IP']/(1.0001-row['dira'])), axis=1)
    print 'added sec vtx dist'

    df['min_track_ip']=df.apply(lambda row: min([row['p0_IP'], row['p1_IP'], row['p2_IP']]), axis=1)
    df['max_track_ip']=df.apply(lambda row: max([row['p0_IP'], row['p1_IP'], row['p2_IP']]), axis=1)

    df['min_track_ipsig']=df.apply(lambda row: min([row['p0_IPSig'], row['p1_IPSig'], row['p2_IPSig']]), axis=1)
    df['max_track_ipsig']=df.apply(lambda row: max([row['p0_IPSig'], row['p1_IPSig'], row['p2_IPSig']]), axis=1)
    
    df['min_DCA']=df.apply(lambda row: min([row['DOCAone'], row['DOCAtwo'], row['DOCAthree']]), axis=1)
    df['max_DCA']=df.apply(lambda row: max([row['DOCAone'], row['DOCAtwo'], row['DOCAthree']]), axis=1)

    df['min_track_chi2']=df.apply(lambda row: min([row['p0_track_Chi2Dof'], row['p1_track_Chi2Dof'], row['p2_track_Chi2Dof']]), axis=1)
    df['max_track_chi2']=df.apply(lambda row: max([row['p0_track_Chi2Dof'], row['p1_track_Chi2Dof'], row['p2_track_Chi2Dof']]), axis=1)

    df['min_dimuon_ip']=df.apply(lambda row: min([row['IP_p0p2'], row['IP_p1p2']]), axis=1)
    df['max_dimuon_ip']=df.apply(lambda row: max([row['IP_p0p2'], row['IP_p1p2']]), axis=1)

    #isolationa to isolationf is pair-wised isolation variable M_iso
    df['min_isolation_a_f']=df.apply(lambda row: min([row['isolationa'], row['isolationb'], row['isolationc'],row['isolationd'], row['isolatione'], row['isolationf']]), axis=1)
    df['max_isolation_a_f']=df.apply(lambda row: max([row['isolationa'], row['isolationb'], row['isolationc'],row['isolationd'], row['isolatione'], row['isolationf']]), axis=1)
    df['sum_isolation_a_f']=df.apply(lambda row: sum([row['isolationa'], row['isolationb'], row['isolationc'],row['isolationd'], row['isolatione'], row['isolationf']]), axis=1)

    df['min_isobdt']=df.apply(lambda row: min([row['p0_IsoBDT'], row['p1_IsoBDT'], row['p2_IsoBDT']]), axis=1)
    df['max_isobdt']=df.apply(lambda row: max([row['p0_IsoBDT'], row['p1_IsoBDT'], row['p2_IsoBDT']]), axis=1)

    df['min_CDF']=df.apply(lambda row: min([row['CDF1'], row['CDF2'], row['CDF3']]), axis=1)
    df['max_CDF']=df.apply(lambda row: max([row['CDF1'], row['CDF2'], row['CDF3']]), axis=1)
    #df['sum_CDF']=df.apply(lambda row: row['CDF1']+row['CDF2']+row['CDF3'], axis=1)

    print 'added min max'
    
    #in Ds preselection part
    #golden feats
    df['sum_sqrt_track_chi2']=df.apply(lambda row: math.sqrt(row['p0_track_Chi2Dof'])+math.sqrt(row['p1_track_Chi2Dof'])+math.sqrt(row['p2_track_Chi2Dof']), axis=1)

    df['sum_track_pt']=df.apply(lambda row: row['p0_pt']+row['p1_pt']+row['p2_pt'], axis=1)
    #df['sum_track_pz']=df.apply(lambda row: row['pz0']+row['pz1']+row['pz2'], axis=1)

    #df['sum_dimuon_ip']=df.apply(lambda row: row['IP_p0p2'] + row['IP_p1p2'], axis=1)
    print 'added sums'

    return df
