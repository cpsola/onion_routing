NETWORKS = {
        "N1": "Seattle01",
        "N2": "Seattle02",
        "N3": "Seattle04",
        "N4": "Seattle08",
        "N5": "Seattle16",
        "N6": "Seattle32"
}

NETWORKS_INV = {v: k for k, v in NETWORKS.items()}

MAX_DURATION_TIME = {
    "zero": {
        "N1": 71436,
        "N2": 70755,
        "N3": 70907,
        "N4": 69432,
        "N5": 68291,
        "N6": 65985},
    "rush": {
        "N1": 46442,
        "N2": 44421,
        "N3": 45310,
        "N4": 42473,
        "N5": 40915,
        "N6": 32070},
}

STARTING_NODES = {
    "zero": [229, 789, 257, 974, 459, 186, 353, 202, 711, 1027, 385, 782, 735, 1002, 689, 716, 899, 562, 329, 317, 932,
             1015, 493, 877, 923, 667, 780, 189, 1097, 929, 798, 214, 40, 1104, 7, 146, 567, 352, 674, 597, 989, 279,
             992, 283, 1136, 981, 413, 916, 1067, 1019, 888, 208, 1025, 488, 797, 147, 635, 207, 719, 1044, 109, 495,
             52, 110, 206, 1178, 277, 640, 256, 40, 87, 155, 480, 706, 500, 878, 406, 808, 621, 1015, 1134, 563, 170,
             1149, 1035, 237, 1136, 910, 145, 83, 768, 552, 306, 1153, 76, 964, 68, 1030, 328, 315],
    "rush": [315, 317, 1045, 388, 808, 636, 996, 896, 954, 871, 587, 700, 788, 731, 857, 528, 629, 910, 61, 677, 963,
             750, 573, 365, 618, 403, 315, 330, 942, 814, 176, 1149, 925, 45, 378, 839, 11, 134, 261, 390, 931, 1143,
             973, 802, 670, 314, 1175, 840, 995, 920, 647, 417, 94, 1051, 584, 1104, 252, 790, 692, 1025, 644, 461, 113,
             478, 777, 98, 576, 457, 910, 672, 237, 1091, 43, 1114, 1059, 981, 1052, 803, 911, 979, 325, 108, 50, 170,
             617, 730, 218, 886, 653, 790, 496, 851, 783, 121, 890, 30, 332, 811, 71, 1027]
}

STARTING_TIMES = {
    "zero": [1847, 1318, 456, 8511, 4728, 1445, 495, 6258, 8287, 3015, 7387, 5356, 1485, 3090, 2433, 4324, 1957, 2199,
             2888, 9355, 2605, 9828, 4783, 4376, 5447, 7463, 8623, 3133, 3033, 2002, 2546, 6503, 2205, 1351, 3687,
             5285, 3098, 8582, 9267, 5894, 3981, 1270, 9819, 7175, 1387, 2818, 1224, 1532, 540, 8076, 8844, 9183, 1602,
             3231, 7822, 4208, 1166, 440, 4657, 3039, 9700, 783, 8247, 4162, 2048, 8666, 6629, 7577, 5399, 292, 9103,
             3756, 6157, 2181, 417, 8996, 2520, 7029, 2420, 1464, 1181, 600, 1533, 6685, 1719, 8662, 9284, 6443, 574,
             6336, 1544, 1237, 1606, 7420, 6117, 8682, 129, 855, 1053, 9867],
    "rush": [30528, 33284, 25106, 29467, 25259, 31724, 28374, 33146, 30516, 32412, 34105, 29698, 27409, 31861, 33345,
             33240, 29405, 26466, 25580, 29267, 29993, 27631, 27884, 28162, 34253, 25496, 34731, 28740, 27677, 28483,
             29187, 28421, 33956, 25045, 33999, 27044, 32124, 32774, 33089, 33339, 32051, 29203, 33372, 26262, 34759,
             27205, 28544, 28922, 33610, 34914, 25909, 25671, 27734, 29721, 25010, 34301, 28920, 33140, 34466, 31872,
             34959, 34921, 26436, 33223, 34219, 33730, 33051, 30666, 25541, 32694, 25701, 27088, 31216, 31316, 28322,
             34277, 25758, 28875, 32627, 28928, 34969, 26009, 31314, 27442, 26730, 28360, 30721, 32328, 31262, 26350,
             29492, 26898, 32912, 31452, 29769, 33098, 28882, 32182, 26526, 30768]
}

ENDING_NODES = {
    "zero" : [969, 791, 1030, 218, 395, 1022, 1053, 1022, 375, 710, 860, 927, 657, 810, 128, 823, 520, 301, 728, 1030,
              895, 113, 253, 137, 915, 1134, 964, 721, 1173, 181, 35, 1175, 765, 1054, 486, 461, 159, 1116, 109, 172,
              205, 307, 877, 76, 183, 838, 952, 760, 600, 591, 581, 82, 207, 1049, 458, 695, 1006, 883, 14, 218, 17,
              1117, 263, 389, 653, 338, 416, 148, 724, 215, 889, 265, 492, 601, 62, 325, 419, 519, 726, 898, 662, 195,
              818, 120, 961, 517, 746, 951, 887, 526, 998, 572, 141, 253, 964, 986, 333, 544, 47, 935],
    "rush" : [994, 961, 470, 325, 554, 1064, 310, 86, 902, 174, 629, 659, 867, 1029, 361, 640, 349, 1145, 1154, 438,
              665, 1125, 649, 282, 16, 88, 843, 363, 403, 756, 669, 960, 498, 724, 355, 973, 284, 40, 179, 849, 185,
              869, 1100, 1079, 452, 37, 839, 237, 596, 148, 633, 928, 688, 304, 70, 30, 23, 116, 299, 716, 1098, 740,
              249, 609, 956, 438, 957, 272, 1082, 169, 347, 428, 226, 324, 234, 920, 771, 1110, 967, 1045, 721, 135,
              14, 87, 1045, 981, 1106, 1031, 759, 470, 1131, 674, 95, 558, 366, 755, 1153, 955, 910, 550]
}
SEP = "---------------------------------"