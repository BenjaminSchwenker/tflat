from yaml import full_load
import numpy as np


def load_config(configFile):
    with open(configFile, "r") as f:
        config = full_load(f)
    return config

def make_display_names(blocks):
    # Map for the variable names
    pretty_name_map = {
        'charge': 'Charge',
        'useCMSFrame(p)': 'p CMS', 'useCMSFrame(cosTheta)': 'cosθ CMS', 'useCMSFrame(phi)': 'phi CMS',
        'electronID': 'e ID', 'muonID': 'μ ID', 'kaonID': 'K ID', 'pionID': 'π ID', 'protonID': 'p ID',
        'pidPairProbabilityExpert(321, 211, CDC)': 'K/π ID CDC',
        'pidPairProbabilityExpert(321, 211, TOP)': 'K/π ID TOP',
        'pidPairProbabilityExpert(321, 211, ARICH)': 'K/π ID ARICH',
        'pidPairProbabilityExpert(11, 211, TOP)': 'e/π ID TOP',
        'pidPairProbabilityExpert(11, 211, ARICH)': 'e/π ID ARICH',
        'pidPairProbabilityExpert(11, 211, ECL)': 'e/π ID ECL',
        'pidPairProbabilityExpert(13, 211, TOP)': 'μ/π ID TOP',
        'pidPairProbabilityExpert(13, 211, ARICH)': 'μ/π ID ARICH',
        'pidPairProbabilityExpert(13, 211, KLM)': 'μ/π ID KLM',
        'pidPairProbabilityExpert(211, 321, TOP)': 'π/K ID TOP',
        'pidPairProbabilityExpert(211, 321, ARICH)': 'π/K ID ARICH',
        'nPXDHits/2': 'nPXD/2', 'nSVDHits/8': 'nSVD/8',
        'dxdiff': 'Δx', 'dydiff': 'Δy', 'dzdiff': 'Δz',
        'clusterEoP': 'Cluster E/p', 'ClusterLAT': 'Cluster LAT',
        'clusterE1E9': 'Cluster E1/E9', 'clusterE9E21': 'Cluster E9/E21',
        'countInList(gamma:tflat)/8': '# gamma/8', 'countInList(pi+:tflat)/6': '# π/6',
        'NumberOfKShortsInRoe': '# K_S', 'ptTracksRoe(TFLATDefaultMask)': 'pT(ROE trk)',
    }

    names = []
    keys = []
    for tag, var_list, n, show_rank in blocks:
        for i in range(1, n + 1):
            for var in var_list:
                label = pretty_name_map.get(var, var)
                names.append(f"{label} ({tag} {i})" if show_rank else f"{label} ({tag})")
                keys.append(f"{label} ({tag})")
    return names, keys

def get_vars(config):
    parameters = config['parameters']
    trk_variable_list = config['trk_variable_list']
    ecl_variable_list = config['ecl_variable_list']
    roe_variable_list = config['roe_variable_list']

    display_names, group_keys = make_display_names([
    ('trk', trk_variable_list, parameters['num_trk'], True),
    ('ecl', ecl_variable_list, parameters['num_ecl'], True),
    ('roe', roe_variable_list, parameters['num_roe'], False),
    ])
    return display_names, group_keys

def get_super_groups():
    # Variable groupped according Sphinx docuementation
    charge_label       = ['Charge (trk)']
    k_trk_label        = ['p CMS (trk)', 'cosθ CMS (trk)', 'phi CMS (trk)']
    k_ecl_label        = ['p CMS (ecl)', 'cosθ CMS (ecl)', 'phi CMS (ecl)']
    pid_trk_label      = ['e ID (trk)', 'μ ID (trk)', 'K ID (trk)', 'π ID (trk)', 'p ID (trk)']
    pid_expert_label   = ['K/π ID CDC (trk)', 'K/π ID TOP (trk)', 'K/π ID ARICH (trk)',
                        'e/π ID TOP (trk)', 'e/π ID ARICH (trk)', 'e/π ID ECL (trk)',
                        'μ/π ID TOP (trk)', 'μ/π ID ARICH (trk)', 'μ/π ID KLM (trk)',
                        'π/K ID TOP (trk)', 'π/K ID ARICH (trk)']
    tracking_trk_label = ['nPXD/2 (trk)', 'nSVD/8 (trk)', 'Δx (trk)', 'Δy (trk)', 'Δz (trk)']
    cluster_label      = ['Cluster E/p (trk)', 'clusterLAT (trk)',
                        'Cluster E1/E9 (ecl)', 'Cluster E9/E21 (ecl)', 'clusterLAT (ecl)']
    roe_vars_label     = ['# gamma/8 (roe)', '# π/6 (roe)', '# K_S (roe)', 'pT(ROE trk) (roe)']

    super_groups = {
        'Charge':          charge_label,
        'Kinematic (trk)': k_trk_label,
        'Kinematic (ecl)': k_ecl_label,
        'PID':             pid_trk_label,
        'PID expert':      pid_expert_label,
        'Tracking':        tracking_trk_label,
        'Cluster':         cluster_label,
        'ROE':             roe_vars_label,
    }
    return super_groups