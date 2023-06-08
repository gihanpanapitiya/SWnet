import pandas as pd
import improve_utils
import os
import urllib
from sklearn.preprocessing import StandardScaler

# ext_genes = ['A1CF', 'ABCC5', 'ABCF1', 'ABHD4', 'ABHD6', 'ABI1', 'ABL1', 'ABL2', 'ACAA1', 'ACAT2', 'ACBD3', 'ACD', 'ACKR3', 'ACLY', 'ACOT9', 'ACSL3', 'ACSL6', 'ACVR1', 'ACVR2A', 'ADAM10', 'ADAT1', 'ADGRE5', 'ADGRG1', 'ADH5', 'ADI1', 'ADO', 'ADRB2', 'AFF1', 'AFF3', 'AFF4', 'AGL', 'AKAP8', 'AKAP8L', 'AKAP9', 'AKR7A2', 'AKT1', 'AKT2', 'AKT3', 'ALAS1', 'ALDH2', 'ALDH7A1', 'ALDOA', 'ALK', 'AMDHD2', 'AMER1', 'ANK1', 'ANKRD10', 'ANO10', 'ANXA7', 'APBB2', 'APOBEC3B', 'APOE', 'APP', 'APPBP2', 'AR', 'ARAF', 'ARFIP2', 'ARHGAP1', 'ARHGAP26', 'ARHGAP5', 'ARHGEF10', 'ARHGEF10L', 'ARHGEF12', 'ARHGEF2', 'ARID1A', 'ARID1B', 'ARID2', 'ARID4B', 'ARID5B', 'ARL4C', 'ARNT', 'ARNT2', 'ARPP19', 'ASAH1', 'ASCC3', 'ASPSCR1', 'ASXL1', 'ASXL2', 'ATF1', 'ATF5', 'ATF6', 'ATG3', 'ATIC', 'ATM', 'ATMIN', 'ATP11B', 'ATP1A1', 'ATP1B1', 'ATP2B3', 'ATP2C1', 'ATP6V0B', 'ATP6V1D', 'ATR', 'ATRX', 'AURKA', 'AURKB', 'AXIN1', 'AXIN2', 'B2M', 'B4GAT1', 'BACE2', 'BAD', 'BAG3', 'BAMBI', 'BAP1', 'BARD1', 'BAX', 'BAZ1A', 'BCL10', 'BCL11A', 'BCL11B', 'BCL2', 'BCL2L12', 'BCL3', 'BCL6', 'BCL7A', 'BCL7B', 'BCL9', 'BCL9L', 'BCLAF1', 'BCOR', 'BCORL1', 'BCR', 'BDH1', 'BECN1', 'BHLHE40', 'BID', 'BIRC2', 'BIRC3', 'BIRC5', 'BIRC6', 'BLCAP', 'BLM', 'BLMH', 'BLVRA', 'BMP4', 'BMP5', 'BMPR1A', 'BNIP3', 'BNIP3L', 'BPHL', 'BRAF', 'BRCA1', 'BRD3', 'BRD4', 'BRIP1', 'BTK', 'BZW2', 'C2CD2', 'C2CD2L', 'C2CD5', 'C5', 'CAB39', 'CACNA1D', 'CALM3', 'CALR', 'CALU', 'CAMSAP2', 'CAMTA1', 'CANT1', 'CAPN1', 'CARD11', 'CARS', 'CASC3', 'CASC5', 'CASK', 'CASP10', 'CASP2', 'CASP3', 'CASP7', 'CASP8', 'CASP9', 'CAST', 'CAT', 'CBFA2T3', 'CBFB', 'CBL', 'CBLB', 'CBLC', 'CBR1', 'CBR3', 'CCDC6', 'CCDC85B', 'CCDC86', 'CCDC92', 'CCL2', 'CCNA1', 'CCNA2', 'CCNB1', 'CCNB1IP1', 'CCNC', 'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CCNE2', 'CCNF', 'CCNH', 'CCP110', 'CCR4', 'CCR7', 'CD209', 'CD274', 'CD28', 'CD320', 'CD40', 'CD44', 'CD74', 'CD79A', 'CD79B', 'CDC20', 'CDC25A', 'CDC25B', 'CDC42', 'CDC45', 'CDC73', 'CDCA4', 'CDH1', 'CDH10', 'CDH11', 'CDH17', 'CDH3', 'CDK1', 'CDK12', 'CDK19', 'CDK2', 'CDK4', 'CDK5R1', 'CDK6', 'CDK7', 'CDKN1B', 'CDKN2A', 'CDKN2C', 'CDX2', 'CEBPD', 'CEBPZ', 'CENPE', 'CEP57', 'CEP89', 'CERK', 'CETN3', 'CFLAR', 'CGRRF1', 'CHAC1', 'CHCHD7', 'CHD2', 'CHD4', 'CHEK1', 'CHEK2', 'CHIC2', 'CHMP6', 'CHN1', 'CHP1', 'CHST11', 'CIAPIN1', 'CIC', 'CIITA', 'CIRBP', 'CISD1', 'CLIC4', 'CLIP1', 'CLP1', 'CLPX', 'CLSTN1', 'CLTB', 'CLTC', 'CLTCL1', 'CNBD1', 'CNBP', 'CNDP2', 'CNOT3', 'CNOT4', 'CNPY3', 'CNTNAP2', 'CNTRL', 'COASY', 'COG2', 'COG4', 'COG7', 'COL1A1', 'COL2A1', 'COL3A1', 'COL4A1', 'COPB2', 'COPS7A', 'CORO1A', 'COX6C', 'CPEB3', 'CPNE3', 'CPSF4', 'CREB1', 'CREB3L1', 'CREB3L2', 'CREBBP', 'CREG1', 'CRELD2', 'CRK', 'CRKL', 'CRLF2', 'CRNKL1', 'CRTAP', 'CRTC1', 'CRTC3', 'CRYZ', 'CSF1R', 'CSF3R', 'CSK', 'CSMD3', 'CSNK1A1', 'CSNK1E', 'CSNK2A2', 'CSRP1', 'CTCF', 'CTNNA2', 'CTNNAL1', 'CTNNB1', 'CTNND1', 'CTNND2', 'CTSL', 'CTTN', 'CUL3', 'CUX1', 'CXCL2', 'CXCR4', 'CYB561', 'CYLD', 'CYP2C8', 'CYSLTR2', 'CYTH1', 'DAG1', 'DAXX', 'DCAF12L2', 'DCC', 'DCK', 'DCTD', 'DCUN1D4', 'DDB2', 'DDIT3', 'DDIT4', 'DDR1', 'DDR2', 'DDX10', 'DDX3X', 'DDX42', 'DDX5', 'DDX6', 'DECR1', 'DEK', 'DENND2D', 'DERA', 'DFFA', 'DFFB', 'DGCR8', 'DHDDS', 'DHRS7', 'DHX29', 'DICER1', 'DLD', 'DMTF1', 'DNAJA3', 'DNAJB1', 'DNAJB2', 'DNAJB6', 'DNAJC15', 'DNM1', 'DNM1L', 'DNM2', 'DNMT1', 'DNMT3A', 'DNTTIP2', 'DPH2', 'DRAP1', 'DROSHA', 'DSG2', 'DUSP11', 'DUSP22', 'DUSP3', 'DUSP4', 'DUSP6', 'DYNLT3', 'DYRK3', 'E2F2', 'EAPP', 'EBF1', 'EBNA1BP2', 'EBP', 'ECD', 'ECH1', 'ECT2L', 'EDEM1', 'EDN1', 'EED', 'EFCAB14', 'EGF', 'EGFR', 'EGR1', 'EIF1AX', 'EIF3E', 'EIF4A2', 'EIF4EBP1', 'EIF5', 'ELAC2', 'ELAVL1', 'ELF3', 'ELF4', 'ELK4', 'ELL', 'ELN', 'ELOVL6', 'EML3', 'EML4', 'ENOPH1', 'ENOSF1', 'EP300', 'EPAS1', 'EPB41L2', 'EPHA3', 'EPHA7', 'EPHB2', 'EPN2', 'EPRS', 'EPS15', 'ERBB2', 'ERBB3', 'ERBB4', 'ERC1', 'ERCC2', 'ERCC3', 'ERCC4', 'ERCC5', 'ERG', 'ERO1A', 'ESR1', 'ETFB', 'ETNK1', 'ETS1', 'ETV1', 'ETV4', 'ETV5', 'ETV6', 'EVL', 'EWSR1', 'EXOSC4', 'EXT1', 'EXT2', 'EZH2', 'EZR', 'FAH', 'FAIM', 'FAM131B', 'FAM135B', 'FAM20B', 'FAM46C', 'FAM47C', 'FANCA', 'FANCC', 'FANCD2', 'FANCE', 'FANCF', 'FANCG', 'FAS', 'FAT1', 'FAT3', 'FAT4', 'FBLN2', 'FBXL12', 'FBXO11', 'FBXO21', 'FBXO7', 'FBXW7', 'FCGR2B', 'FCHO1', 'FCRL4', 'FDFT1', 'FES', 'FEV', 'FEZ2', 'FGFR1', 'FGFR1OP', 'FGFR2', 'FGFR3', 'FGFR4', 'FH', 'FHIT', 'FHL2', 'FIP1L1', 'FIS1', 'FKBP14', 'FKBP4', 'FLI1', 'FLNA', 'FLT3', 'FLT4', 'FNBP1', 'FOS', 'FOSL1', 'FOXA1', 'FOXJ3', 'FOXL2', 'FOXO1', 'FOXO3', 'FOXO4', 'FOXP1', 'FOXR1', 'FPGS', 'FRS2', 'FSD1', 'FSTL3', 'FUBP1', 'FUS', 'FUT1', 'FYN', 'FZD1', 'FZD7', 'G3BP1', 'GAA', 'GABPB1', 'GADD45A', 'GADD45B', 'GALE', 'GAPDH', 'GAS7', 'GATA1', 'GATA2', 'GATA3', 'GDPD5', 'GFOD1', 'GFPT1', 'GHR', 'GLI1', 'GLI2', 'GLOD4', 'GLRX', 'GMNN', 'GNA11', 'GNA15', 'GNAI1', 'GNAI2', 'GNAQ', 'GNAS', 'GNB5', 'GNPDA1', 'GOLGA5', 'GOLT1B', 'GOPC', 'GPATCH8', 'GPC1', 'GPC3', 'GPC5', 'GPER1', 'GPHN', 'GRB10', 'GRB7', 'GRIN2A', 'GRM3', 'GRN', 'GRWD1', 'GSTZ1', 'GTF2A2', 'GTF2E2', 'GTPBP8', 'H2AFV', 'H3F3A', 'H3F3B', 'HACD3', 'HADH', 'HAT1', 'HDAC2', 'HDAC6', 'HEATR1', 'HEBP1', 'HERC6', 'HERPUD1', 'HES1', 'HEY1', 'HIP1', 'HIST1H2BK', 'HIST2H2BE', 'HK1', 'HLA-A', 'HLA-DRA', 'HLF', 'HMG20B', 'HMGA1', 'HMGA2', 'HMGCR', 'HMGCS1', 'HMGN2P46', 'HMOX1', 'HNF1A', 'HNRNPA2B1', 'HOMER2', 'HOOK2', 'HOOK3', 'HOXA11', 'HOXA13', 'HOXA9', 'HOXC11', 'HOXC13', 'HOXD13', 'HPRT1', 'HRAS', 'HS2ST1', 'HSD17B10', 'HSP90AA1', 'HSP90AB1', 'HSPA1A', 'HSPA4', 'HSPA8', 'HSPD1', 'HTATSF1', 'HTRA1', 'HYOU1', 'IARS2', 'ICAM1', 'ICAM3', 'ICMT', 'ID2', 'ID3', 'IDE', 'IDH1', 'IDH2', 'IER3', 'IFNAR1', 'IFRD2', 'IGF1R', 'IGF2BP2', 'IGF2R', 'IGFBP3', 'IGHMBP2', 'IKBKB', 'IKZF1', 'IL13RA1', 'IL1B', 'IL2', 'IL21R', 'IL4R', 'IL6ST', 'IL7R', 'ILK', 'INPP1', 'INPP4B', 'INSIG1', 'INTS3', 'IPO13', 'IQGAP1', 'IRF4', 'IRS4', 'ISOC1', 'ISX', 'ITFG1', 'ITGAE', 'ITGAV', 'ITGB1BP1', 'ITGB5', 'ITK', 'JADE2', 'JAK1', 'JAK2', 'JAK3', 'JAZF1', 'JMJD6', 'JUN', 'KAT6A', 'KAT6B', 'KAT7', 'KCNJ5', 'KCNK1', 'KCTD5', 'KDM3A', 'KDM5A', 'KDM5B', 'KDM5C', 'KDM6A', 'KDR', 'KDSR', 'KEAP1', 'KIAA0100', 'KIAA0355', 'KIAA0753', 'KIAA1549', 'KIF14', 'KIF20A', 'KIF2C', 'KIF5B', 'KIF5C', 'KIT', 'KLF4', 'KLF6', 'KLHDC2', 'KLHL21', 'KLHL9', 'KLK2', 'KMT2A', 'KMT2C', 'KMT2D', 'KNSTRN', 'KRAS', 'KTN1', 'LAGE3', 'LAMA3', 'LAP3', 'LARP4B', 'LASP1', 'LBR', 'LCK', 'LCP1', 'LEF1', 'LEPROTL1', 'LGALS8', 'LGMN', 'LHFP', 'LIFR', 'LIG1', 'LIPA', 'LMNA', 'LMO1', 'LMO2', 'LOXL1', 'LPAR2', 'LPGAT1', 'LPP', 'LRIG3', 'LRP10', 'LRP1B', 'LRPAP1', 'LRRC41', 'LSM14A', 'LSM5', 'LSM6', 'LSR', 'LYL1', 'LYN', 'LYRM1', 'LZTR1', 'MACF1', 'MAF', 'MAFB', 'MALAT1', 'MALT1', 'MAML2', 'MAMLD1', 'MAN2B1', 'MAP2K1', 'MAP2K2', 'MAP2K4', 'MAP2K5', 'MAP3K1', 'MAP3K13', 'MAP3K4', 'MAP4K4', 'MAP7', 'MAPK1', 'MAPK13', 'MAPK1IP1L', 'MAPK9', 'MAPKAPK2', 'MAPKAPK3', 'MAPKAPK5', 'MAST2', 'MAT2A', 'MAX', 'MB21D2', 'MBNL1', 'MBNL2', 'MBOAT7', 'MBTPS1', 'MCM3', 'MCOLN1', 'MCUR1', 'MDM2', 'MDM4', 'ME2', 'MECOM', 'MED12', 'MEF2C', 'MELK', 'MEN1', 'MEST', 'MET', 'MFSD10', 'MGMT', 'MICALL1', 'MIF', 'MITF', 'MKL1', 'MKNK1', 'MLEC', 'MLF1', 'MLH1', 'MLLT1', 'MLLT10', 'MLLT11', 'MLLT3', 'MLLT4', 'MMP1', 'MMP2', 'MN1', 'MNX1', 'MOK', 'MPC2', 'MPL', 'MPZL1', 'MRPL19', 'MRPS16', 'MRPS2', 'MSH2', 'MSH6', 'MSI2', 'MSN', 'MSRA', 'MTA1', 'MTERF3', 'MTF2', 'MTFR1', 'MTHFD2', 'MTOR', 'MUC1', 'MUC16', 'MUC4', 'MUTYH', 'MVP', 'MYB', 'MYBL2', 'MYC', 'MYCBP', 'MYCBP2', 'MYCL', 'MYCN', 'MYD88', 'MYH11', 'MYH9', 'MYL9', 'MYLK', 'MYO10', 'MYO5A', 'MYOD1', 'N4BP2', 'NAB2', 'NACA', 'NBEA', 'NBN', 'NCAPD2', 'NCK2', 'NCKIPSD', 'NCOA1', 'NCOA2', 'NCOA3', 'NCOR1', 'NCOR2', 'NDRG1', 'NENF', 'NET1', 'NF1', 'NF2', 'NFATC2', 'NFATC3', 'NFATC4', 'NFE2L2', 'NFIB', 'NFIL3', 'NFKB2', 'NFKBIA', 'NFKBIB', 'NFKBIE', 'NIN', 'NIPSNAP1', 'NISCH', 'NIT1', 'NMT1', 'NNT', 'NOL3', 'NOLC1', 'NONO', 'NOS3', 'NOSIP', 'NOTCH1', 'NOTCH2', 'NPC1', 'NPDC1', 'NPEPL1', 'NPM1', 'NPRL2', 'NR1H2', 'NR2F6', 'NR3C1', 'NR4A3', 'NRAS', 'NRG1', 'NRIP1', 'NSD1', 'NSDHL', 'NT5C2', 'NT5DC2', 'NTHL1', 'NTRK1', 'NTRK3', 'NUCB2', 'NUDCD3', 'NUDT9', 'NUMA1', 'NUP133', 'NUP214', 'NUP85', 'NUP88', 'NUP93', 'NUP98', 'NUSAP1', 'NUTM1', 'NVL', 'OLIG2', 'OMD', 'ORC1', 'OXA1L', 'OXCT1', 'OXSR1', 'P2RY8', 'P4HA2', 'P4HTM', 'PACSIN3', 'PAF1', 'PAFAH1B1', 'PAFAH1B2', 'PAFAH1B3', 'PAICS', 'PAK1', 'PAK4', 'PAK6', 'PALB2', 'PAN2', 'PARP1', 'PARP2', 'PAX3', 'PAX5', 'PAX7', 'PAX8', 'PBRM1', 'PBX1', 'PCBD1', 'PCBP1', 'PCCB', 'PCK2', 'PCM1', 'PCMT1', 'PCNA', 'PDCD1LG2', 'PDE4DIP', 'PDGFA', 'PDGFB', 'PDGFRA', 'PDGFRB', 'PDHX', 'PDIA5', 'PDLIM1', 'PDS5A', 'PECR', 'PER1', 'PEX11A', 'PFKL', 'PGM1', 'PGRMC1', 'PHF6', 'PHGDH', 'PHKA1', 'PHKB', 'PHKG2', 'PHOX2B', 'PICALM', 'PIGB', 'PIH1D1', 'PIK3C2B', 'PIK3C3', 'PIK3CA', 'PIK3CB', 'PIK3R1', 'PIK3R3', 'PIK3R4', 'PIM1', 'PIN1', 'PKIG', 'PLA2G15', 'PLA2G4A', 'PLAG1', 'PLCB3', 'PLCG1', 'PLEKHJ1', 'PLEKHM1', 'PLK1', 'PLOD3', 'PLP2', 'PLS1', 'PLSCR1', 'PMAIP1', 'PML', 'PMM2', 'PMS1', 'PMS2', 'PNKP', 'PNP', 'POLB', 'POLD1', 'POLE', 'POLE2', 'POLG', 'POLG2', 'POLQ', 'POLR1C', 'POLR2I', 'POLR2K', 'POP4', 'POT1', 'POU2AF1', 'POU5F1', 'PPARD', 'PPARG', 'PPFIBP1', 'PPIC', 'PPIE', 'PPM1D', 'PPOX', 'PPP1R13B', 'PPP2R1A', 'PPP2R3C', 'PPP2R5A', 'PPP2R5E', 'PPP6C', 'PRAF2', 'PRCC', 'PRCP', 'PRDM1', 'PRDM16', 'PRDM2', 'PREX2', 'PRF1', 'PRKACA', 'PRKAG2', 'PRKAR1A', 'PRKCB', 'PRKCD', 'PRKCQ', 'PRKX', 'PROS1', 'PRPF4', 'PRPF40B', 'PRR15L', 'PRR7', 'PRRX1', 'PRSS23', 'PRUNE1', 'PSIP1', 'PSMB8', 'PSMD10', 'PSMD4', 'PSME1', 'PSMF1', 'PSMG1', 'PSRC1', 'PTCH1', 'PTEN', 'PTGS2', 'PTK2', 'PTK2B', 'PTK6', 'PTPN1', 'PTPN11', 'PTPN12', 'PTPN13', 'PTPN6', 'PTPRB', 'PTPRC', 'PTPRD', 'PTPRF', 'PTPRK', 'PTPRT', 'PUF60', 'PWP1', 'PWWP2A', 'PXN', 'PYCR1', 'PYGL', 'QKI', 'RAB11FIP2', 'RAB21', 'RAB27A', 'RAB31', 'RAB4A', 'RABEP1', 'RAC1', 'RAD17', 'RAD21', 'RAD51B', 'RAD51C', 'RAD9A', 'RAE1', 'RAF1', 'RAI14', 'RALA', 'RALB', 'RALGDS', 'RANBP2', 'RAP1GAP', 'RAP1GDS1', 'RARA', 'RASA1', 'RB1', 'RBKS', 'RBM10', 'RBM15', 'RBM6', 'RECQL4', 'REEP5', 'REL', 'RELB', 'RET', 'RFC2', 'RFC5', 'RFNG', 'RFWD3', 'RFX5', 'RGS2', 'RGS7', 'RHEB', 'RHOA', 'RHOH', 'RMI2', 'RNF167', 'RNF213', 'RNF43', 'RNH1', 'RNMT', 'RNPS1', 'ROBO2', 'ROS1', 'RPA1', 'RPA2', 'RPA3', 'RPIA', 'RPL10', 'RPL22', 'RPL39L', 'RPL5', 'RPN1', 'RPS5', 'RPS6', 'RPS6KA1', 'RRAGA', 'RRP12', 'RRP1B', 'RRP8', 'RRS1', 'RSPO2', 'RSU1', 'RTN2', 'RUNX1', 'RUNX1T1', 'RUVBL1', 'S100A13', 'S100A4', 'S100A7', 'SACM1L', 'SALL4', 'SBDS', 'SCAND1', 'SCARB1', 'SCCPDH', 'SCP2', 'SCRN1', 'SCYL3', 'SDC4', 'SDHA', 'SDHB', 'SDHC', 'SENP6', 'SERPINE1', 'SESN1', 'SET', 'SETBP1', 'SETD1B', 'SETD2', 'SF3B1', 'SFN', 'SFPQ', 'SFRP4', 'SGCB', 'SGK1', 'SH2B3', 'SH3BP5', 'SH3GL1', 'SHC1', 'SIRPA', 'SIRT3', 'SIX1', 'SIX2', 'SKI', 'SKIV2L', 'SKP1', 'SLC11A2', 'SLC1A4', 'SLC25A13', 'SLC25A14', 'SLC25A4', 'SLC25A46', 'SLC27A3', 'SLC2A6', 'SLC34A2', 'SLC35A1', 'SLC35A3', 'SLC35B1', 'SLC35F2', 'SLC37A4', 'SLC45A3', 'SLC5A6', 'SMAD2', 'SMAD3', 'SMAD4', 'SMARCA4', 'SMARCB1', 'SMARCC1', 'SMARCD1', 'SMARCD2', 'SMARCE1', 'SMC1A', 'SMC3', 'SMC4', 'SMNDC1', 'SMO', 'SNAP25', 'SNCA', 'SND1', 'SNX11', 'SNX13', 'SNX6', 'SNX7', 'SOCS1', 'SOCS2', 'SORBS3', 'SOX21', 'SOX4', 'SPAG4', 'SPAG7', 'SPDEF', 'SPEN', 'SPOP', 'SPP1', 'SPR', 'SPRED2', 'SPTAN1', 'SPTLC2', 'SQSTM1', 'SRC', 'SRGAP3', 'SRSF2', 'SS18', 'SS18L1', 'SSBP2', 'SSX1', 'ST3GAL5', 'ST6GALNAC2', 'ST7', 'STAG1', 'STAG2', 'STAMBP', 'STAP2', 'STAT1', 'STAT3', 'STAT5B', 'STAT6', 'STIL', 'STK10', 'STK11', 'STK25', 'STMN1', 'STRN', 'STX1A', 'STX4', 'STXBP1', 'STXBP2', 'SUFU', 'SUPV3L1', 'SUV39H1', 'SUZ12', 'SYK', 'SYNE2', 'SYNGR3', 'SYPL1', 'TAL1', 'TAL2', 'TARBP1', 'TATDN2', 'TBC1D31', 'TBC1D9B', 'TBL1XR1', 'TBP', 'TBPL1', 'TBX2', 'TBX3', 'TBXA2R', 'TCEA1', 'TCEA2', 'TCEAL4', 'TCERG1', 'TCF12', 'TCF3', 'TCF7L2', 'TCFL5', 'TCL1A', 'TCTA', 'TCTN1', 'TEC', 'TERF2IP', 'TERT', 'TES', 'TESK1', 'TET1', 'TET2', 'TEX10', 'TFAP2A', 'TFDP1', 'TFE3', 'TFEB', 'TFG', 'TFPT', 'TFRC', 'TGFB3', 'TGFBR2', 'THAP11', 'THRAP3', 'TIAM1', 'TICAM1', 'TIMELESS', 'TIMM17B', 'TIMM22', 'TIMM9', 'TIMP2', 'TIPARP', 'TJP1', 'TLE1', 'TLK2', 'TLR4', 'TLX1', 'TLX3', 'TM9SF2', 'TM9SF3', 'TMCO1', 'TMED10', 'TMEM109', 'TMEM127', 'TMEM50A', 'TMEM97', 'TMPRSS2', 'TNC', 'TNFAIP3', 'TNFRSF14', 'TNFRSF17', 'TNFRSF21', 'TNIP1', 'TOMM34', 'TOMM70', 'TOP1', 'TOP2A', 'TOPBP1', 'TOR1A', 'TP53', 'TP53BP1', 'TP53BP2', 'TP63', 'TPD52L2', 'TPM1', 'TPM3', 'TPM4', 'TPR', 'TRAF7', 'TRAK2', 'TRAM2', 'TRAP1', 'TRAPPC3', 'TRAPPC6A', 'TRIB1', 'TRIB3', 'TRIM13', 'TRIM2', 'TRIM24', 'TRIM27', 'TRIM33', 'TRIP11', 'TRRAP', 'TSC1', 'TSC2', 'TSC22D3', 'TSEN2', 'TSHR', 'TSKU', 'TSPAN3', 'TSPAN4', 'TSPAN6', 'TSTA3', 'TUBB6', 'TXLNA', 'TXNDC9', 'TXNL4B', 'TXNRD1', 'U2AF1', 'UBE2A', 'UBE2C', 'UBE2J1', 'UBE2L6', 'UBE3B', 'UBE3C', 'UBQLN2', 'UBR5', 'UBR7', 'UFM1', 'UGDH', 'USP1', 'USP14', 'USP22', 'USP44', 'USP6', 'USP6NL', 'USP7', 'USP8', 'UTP14A', 'VAPB', 'VAT1', 'VAV1', 'VAV3', 'VGLL4', 'VHL', 'VPS28', 'VPS72', 'VTI1A', 'WAS', 'WASF3', 'WDR61', 'WDR7', 'WDTC1', 'WFS1', 'WHSC1', 'WHSC1L1', 'WIF1', 'WIPF2', 'WNK2', 'WRN', 'WT1', 'WWTR1', 'XBP1', 'XPA', 'XPC', 'XPNPEP1', 'XPO1', 'XPO7', 'YKT6', 'YME1L1', 'YTHDF1', 'YWHAE', 'ZBTB16', 'ZCCHC8', 'ZDHHC6', 'ZEB1', 'ZFHX3', 'ZFP36', 'ZMIZ1', 'ZMYM2', 'ZMYM3', 'ZNF131', 'ZNF274', 'ZNF318', 'ZNF331', 'ZNF384', 'ZNF395', 'ZNF429', 'ZNF451', 'ZNF479', 'ZNF521', 'ZNF586', 'ZNF589', 'ZNRF3', 'ZRSR2']



def get_drug_response_data(df, metric):
    
    # df = rs_train.copy()
    smiles_df = improve_utils.load_smiles_data()
    data_smiles_df = pd.merge(df, smiles_df, on = "improve_chem_id", how='left') 
    data_smiles_df = data_smiles_df.dropna(subset=[metric])
    data_smiles_df = data_smiles_df[['improve_sample_id', 'smiles', 'improve_chem_id', metric]]
    data_smiles_df = data_smiles_df.drop_duplicates()
    data_smiles_df = data_smiles_df.reset_index(drop=True)

    return data_smiles_df


def sava_split_files(df, file_name, metric='ic50'):

    tmp = df[['improve_sample_id', 'improve_chem_id', metric]]
    tmp = tmp.rename(columns={'improve_sample_id':'cell_line_id',
                        'improve_chem_id':'drug_id',
                        metric:'labels'})
    tmp.to_csv(file_name, index=False)

def candle_preprocess(data_type='CCLE', 
                         metric='ic50', 
                         data_path=None,
                         ext_gene_file=None

):
        
    # data_type='CCLE'
    # metric='ic50'
    rs_all = improve_utils.load_single_drug_response_data(source=data_type, split=0,
                                                        split_type=["train", "test", 'val'],
                                                        y_col_name=metric)

    rs_train = improve_utils.load_single_drug_response_data(source=data_type,
                                                            split=0, split_type=["train"],
                                                            y_col_name=metric)
    rs_test = improve_utils.load_single_drug_response_data(source=data_type,
                                                        split=0,
                                                        split_type=["test"],
                                                        y_col_name=metric)
    rs_val = improve_utils.load_single_drug_response_data(source=data_type,
                                                        split=0,
                                                        split_type=["val"],
                                                        y_col_name=metric)



    train_df = get_drug_response_data(rs_train, metric)
    val_df = get_drug_response_data(rs_val, metric)
    test_df = get_drug_response_data(rs_test, metric)

    
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    all_df.reset_index(drop=True, inplace=True)

    sava_split_files(train_df, data_path+'/CCLE/CCLE_Data/train.csv', metric)
    sava_split_files(val_df, data_path+'/CCLE/CCLE_Data/val.csv', metric)
    sava_split_files(test_df, data_path+'/CCLE/CCLE_Data/test.csv', metric)



    smi_candle = all_df[['improve_chem_id', 'smiles']]
    smi_candle.drop_duplicates(inplace=True)
    smi_candle.reset_index(drop=True, inplace=True)
    smi_candle.set_index('improve_chem_id', inplace=True)
    smi_candle.index.name=None
    smi_candle.to_csv(data_path+'/CCLE/CCLE_Data/CCLE_smiles.csv')

    mutation_data = improve_utils.load_cell_mutation_data(gene_system_identifier="Entrez")
    expr_data = improve_utils.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
    mutation_data = mutation_data.reset_index()
    gene_data = mutation_data.columns[1:]

    ext_genes = pd.read_csv(ext_gene_file,index_col=0)
    common_genes=list(set(gene_data).intersection(ext_genes.columns))
    # common_genes=list(set(gene_data).intersection(ext_genes))

    mut = mutation_data[mutation_data.improve_sample_id.isin(all_df.improve_sample_id)]
    

    mut = mut.loc[:, ['improve_sample_id'] + common_genes ]
    mut.improve_sample_id.nunique() == mut.shape[0]
    mut.set_index('improve_sample_id', inplace=True)
    mut = mut.gt(0).astype(int)

    expr = expr_data[expr_data.index.isin(mut.index)]
    expr = expr.loc[:, common_genes]

    sc = StandardScaler()

    expr[:] = sc.fit_transform(expr[:])


    expr.index.name=None
    expr.to_csv(data_path+'/CCLE/CCLE_Data/CCLE_RNAseq.csv')

    mut.index.name=None
    mut.to_csv(data_path+'/CCLE/CCLE_Data/CCLE_DepMap.csv')

    all_df=all_df[['improve_sample_id', 'improve_chem_id', metric]]
    all_df=all_df.rename(columns={'improve_sample_id':'cell_line_id',
                        'improve_chem_id':'drug_id',
                        metric:'labels'})
    all_df.to_csv(data_path+'/CCLE/CCLE_Data/CCLE_cell_drug_labels.csv', index=False)