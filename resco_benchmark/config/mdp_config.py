mdp_configs = {
    'FMA2CFull': {
        'grid4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['A0', 'B0', 'B1', 'A1'],
                'bot_right_mgr': ['C0', 'D0', 'D1', 'C1'],
                'top_right_mgr': ['C2', 'D2', 'D3', 'C3'],
                'top_left_mgr': ['A2', 'B2', 'B3', 'A3']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'arterial4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['nt1', 'nt2', 'nt6', 'nt5'],
                'bot_right_mgr': ['nt3', 'nt4', 'nt8', 'nt7'],
                'top_right_mgr': ['nt11', 'nt12', 'nt16', 'nt15'],
                'top_left_mgr': ['nt9', 'nt10', 'nt14', 'nt13']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'ingolstadt7': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['cluster_1757124350_1757124352', 'gneJ143', 'gneJ207',
                            'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190'],
                'bot_mgr': ['32564122', 'gneJ260', 'gneJ210']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'ingolstadt21': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['cluster_1757124350_1757124352', 'gneJ143', 'gneJ207',
                            'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190'],
                'bot_mgr': ['32564122', 'gneJ255', 'gneJ210'],
                'bot_left_mgr': ['243351999', '89173808', '89173763'],
                'top_left_mgr': ['cluster_1863241547_1863241548_1976170214', '1863241632', '2330725114', 'gneJ208',
                                 '243749571'],
                'top_right_mgr': ['243641585', 'gneJ257', '30503246', '30624898', '89127267',
                                  'cluster_1427494838_273472399']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr', 'top_right_mgr'],
                'bot_mgr': ['top_mgr', 'bot_left_mgr'],
                'bot_left_mgr': ['bot_mgr', 'top_left_mgr'],
                'top_left_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['top_mgr', 'top_left_mgr']
            }
        },
        'cologne8': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['247379907', '256201389', '26110729', '280120513', '62426694'],
                'bot_mgr': ['32319828', '252017285', 'cluster_1098574052_1098574061_247379905']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'ksu_map': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'north_mgr': ['11556036914', '448837014', '461883453', '461883473', '462020073', '462020075', '473871827', '5479208872', '5479208874', '5481716880', '6682007237', '6749481529', '6749481539', '6749481540', '6749481542', '696590144', '696590146'],
                'south_mgr': ['11558029064', '1472529962', '1472530067', '1472530069', '472510383', '9990246809'],
                'east_mgr': ['11558029064', '461883453', '461883473', '462020073', '462020075', '472510383', '473871827', '6682007237', '6749481529', '6749481539', '6749481540', '6749481542', '696590144', '696590146', '9990246809'],
                'west_mgr': ['11556036914', '1472529962', '1472530067', '1472530069', '448837014', '5479208872', '5479208874', '5481716880']
            },
           'management_neighbors': {
                'north_mgr': ['south_mgr', 'east_mgr', 'west_mgr'],
                'south_mgr': ['north_mgr', 'east_mgr', 'west_mgr'],
                'east_mgr': ['north_mgr', 'south_mgr', 'west_mgr'],
                'west_mgr': ['north_mgr', 'south_mgr', 'east_mgr']
            }
        },
    },
    'FMA2C': {
        'grid4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['A0', 'B0', 'B1', 'A1'],
                'bot_right_mgr': ['C0', 'D0', 'D1', 'C1'],
                'top_right_mgr': ['C2', 'D2', 'D3', 'C3'],
                'top_left_mgr': ['A2', 'B2', 'B3', 'A3']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'arterial4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['nt1', 'nt2', 'nt6', 'nt5'],
                'bot_right_mgr': ['nt3', 'nt4', 'nt8', 'nt7'],
                'top_right_mgr': ['nt11', 'nt12', 'nt16', 'nt15'],
                'top_left_mgr': ['nt9', 'nt10', 'nt14', 'nt13']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'ingolstadt1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['gneJ207']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        'ingolstadt7': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['cluster_1757124350_1757124352', 'gneJ143', 'gneJ207', 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190'],
                'bot_mgr': ['32564122', 'gneJ260', 'gneJ210']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'ingolstadt21': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['cluster_1757124350_1757124352', 'gneJ143', 'gneJ207', 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190'],
                'bot_mgr': ['32564122', 'gneJ255', 'gneJ210'],
                'bot_left_mgr': ['243351999', '89173808', '89173763'],
                'top_left_mgr': ['cluster_1863241547_1863241548_1976170214', '1863241632', '2330725114', 'gneJ208', '243749571'],
                'top_right_mgr': ['243641585', 'gneJ257', '30503246', '30624898', '89127267', 'cluster_1427494838_273472399']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr', 'top_right_mgr'],
                'bot_mgr': ['top_mgr', 'bot_left_mgr'],
                'bot_left_mgr': ['bot_mgr', 'top_left_mgr'],
                'top_left_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['top_mgr', 'top_left_mgr']
            }
        },
        'cologne3': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['360082', '360086'],
                'bot_mgr': ['GS_cluster_2415878664_254486231_359566_359576']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne8': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['247379907', '256201389', '26110729', '280120513', '62426694'],
                'bot_mgr': ['32319828', '252017285', 'cluster_1098574052_1098574061_247379905']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['GS_cluster_357187_359543']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        'ksu_map': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'north_mgr': ['11556036914', '448837014', '461883453', '461883473', '462020073', '462020075', '473871827', '5479208872', '5479208874', '5481716880', '6682007237', '6749481529', '6749481539', '6749481540', '6749481542', '696590144', '696590146'],
                'south_mgr': ['11558029064', '1472529962', '1472530067', '1472530069', '472510383', '9990246809'],
                'east_mgr': ['11558029064', '461883453', '461883473', '462020073', '462020075', '472510383', '473871827', '6682007237', '6749481529', '6749481539', '6749481540', '6749481542', '696590144', '696590146', '9990246809'],
                'west_mgr': ['11556036914', '1472529962', '1472530067', '1472530069', '448837014', '5479208872', '5479208874', '5481716880']
            },
           'management_neighbors': {
                'north_mgr': ['south_mgr', 'east_mgr', 'west_mgr'],
                'south_mgr': ['north_mgr', 'east_mgr', 'west_mgr'],
                'east_mgr': ['north_mgr', 'south_mgr', 'west_mgr'],
                'west_mgr': ['north_mgr', 'south_mgr', 'east_mgr']
            }
        },
    },
    'FMA2CVAL': {
        'grid4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['A0', 'B0', 'B1', 'A1'],
                'bot_right_mgr': ['C0', 'D0', 'D1', 'C1'],
                'top_right_mgr': ['C2', 'D2', 'D3', 'C3'],
                'top_left_mgr': ['A2', 'B2', 'B3', 'A3']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'arterial4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['nt1', 'nt2', 'nt6', 'nt5'],
                'bot_right_mgr': ['nt3', 'nt4', 'nt8', 'nt7'],
                'top_right_mgr': ['nt11', 'nt12', 'nt16', 'nt15'],
                'top_left_mgr': ['nt9', 'nt10', 'nt14', 'nt13']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'ingolstadt1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['gneJ207']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        'ingolstadt7': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['cluster_1757124350_1757124352', 'gneJ143', 'gneJ207', 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190'],
                'bot_mgr': ['32564122', 'gneJ260', 'gneJ210']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'ingolstadt21': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['cluster_1757124350_1757124352', 'gneJ143', 'gneJ207', 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190'],
                'bot_mgr': ['32564122', 'gneJ255', 'gneJ210'],
                'bot_left_mgr': ['243351999', '89173808', '89173763'],
                'top_left_mgr': ['cluster_1863241547_1863241548_1976170214', '1863241632', '2330725114', 'gneJ208', '243749571'],
                'top_right_mgr': ['243641585', 'gneJ257', '30503246', '30624898', '89127267', 'cluster_1427494838_273472399']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr', 'top_right_mgr'],
                'bot_mgr': ['top_mgr', 'bot_left_mgr'],
                'bot_left_mgr': ['bot_mgr', 'top_left_mgr'],
                'top_left_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['top_mgr', 'top_left_mgr']
            }
        },
        'cologne3': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['360082', '360086'],
                'bot_mgr': ['GS_cluster_2415878664_254486231_359566_359576']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne8': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['247379907', '256201389', '26110729', '280120513', '62426694'],
                'bot_mgr': ['32319828', '252017285', 'cluster_1098574052_1098574061_247379905']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['GS_cluster_357187_359543']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        'ksu_map': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'north_mgr': ['11556036914', '448837014', '461883453', '461883473', '462020073', '462020075', '473871827', '5479208872', '5479208874', '5481716880', '6682007237', '6749481529', '6749481539', '6749481540', '6749481542', '696590144', '696590146'],
                'south_mgr': ['11558029064', '1472529962', '1472530067', '1472530069', '472510383', '9990246809'],
                'east_mgr': ['11558029064', '461883453', '461883473', '462020073', '462020075', '472510383', '473871827', '6682007237', '6749481529', '6749481539', '6749481540', '6749481542', '696590144', '696590146', '9990246809'],
                'west_mgr': ['11556036914', '1472529962', '1472530067', '1472530069', '448837014', '5479208872', '5479208874', '5481716880']
            },
           'management_neighbors': {
                'north_mgr': ['south_mgr', 'east_mgr', 'west_mgr'],
                'south_mgr': ['north_mgr', 'east_mgr', 'west_mgr'],
                'east_mgr': ['north_mgr', 'south_mgr', 'west_mgr'],
                'west_mgr': ['north_mgr', 'south_mgr', 'east_mgr']
            }
        },
    },
    'MA2C': {
        'grid4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 2.0,
            'clip_wait': 2.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
        },
        'arterial4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 2.0,
            'clip_wait': 2.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
        },
        'ingolstadt7': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 2.0,
            'clip_wait': 2.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
        },
        'ingolstadt21': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 2.0,
            'clip_wait': 2.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
        },
        'cologne8': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 2.0,
            'clip_wait': 2.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
        },
        'ingolstadt1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
        },
        'cologne1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
        },
        'cologne3': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
        },
        'ksu_map': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 2.0,
            'clip_wait': 2.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
        }
    }
}
