import pandas as pd

# Question 2 a) and 3 a) Training Data:

# (( 1.7199393288855, 82.401402977991, 33), 'M' )
# (( 1.7417125437108, 83.747401859142, 31), 'M' )
# (( 1.5471739240174, 64.544493914125, 32), 'W' )
# (( 1.6883507370778, 73.503936043062, 26), 'W' )
# (( 1.7230486340695, 83.108315327803, 32), 'M' )
# (( 1.7195436481313, 77.275734890275, 27), 'M' )
# (( 1.5759318307814, 74.322865835104, 33), 'M' )
# (( 1.7399768544776, 74.08611050345, 30), 'M' )
# (( 1.5541388059382, 69.981865975555, 33), 'W' )
# (( 1.6645585608241, 67.716271424719, 25), 'W' )
# (( 1.6880491048666, 76.699703836144, 34), 'W' )
# (( 1.7611117325978, 76.597218154687, 28), 'W' )
# (( 1.6238364197393, 66.54810194208, 24), 'W' )
# (( 1.5979508998164, 59.718334372037, 32), 'W' )


# Question 2 a) and 3 a) Test Data:

# ( 1.8184923035897, 66.646602087329, 29)
# ( 1.648406974399, 73.059070491065, 34)
# ( 1.6474627731456, 70.909032272466, 29)
# ( 1.7661393134637, 79.730137027542, 20)


Data = [
    (1.702638845375, 76.8027458069, 25, "M"),
    (1.6971990153961, 77.269871641804, 24, "W"),
    (1.8346274304775, 83.11025421252, 23, "M"),
    (1.9370703002624, 81.158629828346, 30, "M"),
    (1.8833449771323, 79.56130579047, 29, "M"),
    (1.7708625844432, 79.874386018085, 32, "M"),
    (1.7663028282995, 84.502366446944, 31, "M"),
    (1.8589201835232, 74.222687220586, 27, "M"),
    (1.6885861305035, 85.162952063685, 31, "M"),
    (1.8535584641831, 83.137510438721, 28, "W"),
    (1.9796834737297, 89.959254130863, 30, "M"),
    (1.8298612400188, 82.203525511581, 26, "W"),
    (1.8624273996642, 86.337327617365, 35, "W"),
    (1.7827230671223, 78.196155271048, 29, "W"),
    (1.7097182035075, 81.856716442489, 30, "W"),
    (1.8033598936309, 88.528072627768, 30, "M"),
    (1.6685071195502, 77.666072395403, 27, "W"),
    (1.7883749923733, 78.061393206212, 31, "W"),
    (1.5774249383703, 74.280573621626, 24, "W"),
    (1.6715541277347, 84.663993334148, 27, "W"),
    (1.7061901404023, 75.046986361069, 25, "'W'"),
    (1.6982617822049, 86.34910798392, 25, "M"),
    (1.7699522417883, 72.378754618233, 27, "W"),
    (1.8074555498345, 81.563385035104, 34, "M"),
    (1.7336344130993, 84.789650310141, 28, "W"),
    (1.7637559408923, 82.519046384647, 33, "W"),
    (1.9612373318942, 79.264595196285, 31, "W"),
    (1.7006740147337, 75.044274701568, 32, "W"),
    (1.8741476211353, 85.25212528088, 31, "M"),
    (1.6668225595969, 77.220144882458, 29, "W"),
    (1.6674410419192, 82.189644469724, 35, "M"),
    (1.8192517926059, 84.55450892662, 26, "M"),
    (1.803888487877, 81.623692145317, 31, "W"),
    (1.7989567016259, 83.40314694162, 28, "W"),
    (1.6674184320267, 74.992833970321, 28, "W"),
    (1.9440262777634, 93.580098836386, 30, "M"),
    (1.792299479301, 78.655236623035, 25, "M"),
    (1.8491896144086, 82.364749117518, 26, "W"),
    (1.6790980651054, 82.020501853106, 33, "W"),
    (1.8108311807972, 80.353545859227, 29, "M"),
    (1.6851539781568, 75.471531790552, 31, "W"),
    (1.7491952654594, 88.437615156479, 35, "M"),
    (1.8065734415125, 82.362829683126, 22, "W"),
    (1.8770874605189, 84.093265377515, 34, "W"),
    (1.9425917648313, 89.583619267718, 35, "M"),
    (1.6772375272977, 76.196383299684, 33, "M"),
    (1.8133787305005, 88.534411367357, 25, "M"),
    (1.9520946131297, 90.60816451055, 29, "M"),
    (1.7105124996836, 74.244259515333, 29, "W"),
    (1.8747129772801, 81.501337268202, 31, "W"),
    (1.8270655024764, 76.324793547757, 26, "W"),
    (1.7408230611569, 87.986280696261, 28, "M"),
    (2.0197945491603, 90.244963830258, 28, "M"),
    (1.7969513517324, 83.750706953145, 35, "W"),
    (1.7592161138534, 92.175911112026, 24, "M"),
    (1.8590039822747, 87.996673936472, 27, "M"),
    (1.8266498611481, 72.953276755922, 29, "W"),
    (1.5529634283349, 80.245162164437, 23, "W"),
    (1.7814021683245, 75.713602798286, 36, "M"),
    (1.9099255188094, 86.266289434563, 31, "M"),
    (1.633935991765, 75.404183385482, 32, "W"),
    (1.7273073187477, 82.988647109791, 26, "W"),
    (1.7964236924664, 82.11017678285, 27, "M"),
    (1.833390618975, 89.344442788666, 30, "M"),
    (1.7770839758536, 80.25659042222, 29, "W"),
    (1.8102584952947, 83.596052995276, 27, "M"),
    (1.8918439935374, 79.767829367771, 29, "W"),
    (1.7356653962159, 76.973268737453, 35, "M"),
    (1.8421586081466, 83.645790877883, 31, "M"),
    (1.6999928662615, 79.345943760159, 33, "W"),
    (1.8053947620229, 75.671819957746, 27, "W"),
    (1.7992623664176, 85.555169387001, 32, "W"),
    (1.8429583992571, 88.801240622086, 27, "W"),
    (1.6760767539001, 80.070362562453, 32, "W"),
    (1.7062600310352, 74.122181260759, 25, "W"),
    (1.8246212380646, 83.905608629244, 29, "W"),
    (1.8614239202489, 82.627539969869, 34, "M"),
    (1.8793995820646, 83.226992105366, 26, "W"),
    (1.8713262435137, 79.413841051907, 34, "W"),
    (1.8205271368966, 81.512066427738, 40, "M"),
    (1.7717280224846, 84.736232373479, 37, "W"),
    (1.8951950464043, 86.300296360543, 31, "M"),
    (1.9375787936641, 77.167149804093, 30, "M"),
    (1.7595879962211, 82.686218847023, 29, "M"),
    (1.8119168196927, 83.652787701886, 28, "M"),
    (1.9653640113283, 86.262710292413, 31, "M"),
    (1.8506759438855, 86.960833962812, 27, "M"),
    (1.9662825741207, 85.552578692745, 37, "M"),
    (1.7739827741358, 79.642015331219, 29, "W"),
    (1.9000620015957, 91.090231366632, 25, "M"),
    (1.8083321491638, 90.577992691668, 33, "M"),
    (1.8926255422199, 76.458690236945, 23, "W"),
    (1.9423909283884, 94.784788042942, 31, "M"),
    (1.6820354972643, 72.618517979774, 28, "W"),
    (1.8810428176045, 87.473840166996, 38, "W"),
    (1.7925050685942, 75.305458771862, 34, "W"),
    (1.7135333161607, 73.429952021307, 29, "W"),
    (1.6929981504899, 78.575801121145, 30, "W"),
    (1.8385070463272, 86.287090707398, 27, "M"),
    (1.9548383697676, 85.839734215693, 27, "M"),
    (1.8003075637533, 75.019501153414, 28, "W"),
    (1.9063189434246, 79.934613436695, 27, "W"),
    (1.8919255011921, 88.178802078187, 32, "W"),
    (1.6970865397842, 77.453411495327, 29, "M"),
    (1.7989209291624, 81.784528286328, 29, "M"),
    (1.749636086247, 81.512257371917, 30, "M"),
    (2.0114844828231, 94.448590274185, 27, "M"),
    (1.9032833564153, 90.140224495595, 29, "M"),
    (1.8756711011311, 90.28107807806, 29, "M"),
    (1.6834673751784, 76.949406295267, 34, "M"),
    (1.7893653152244, 81.821186112406, 27, "M"),
    (1.7606970701332, 87.774843787233, 26, "M"),
    (1.7277743527983, 82.169604342142, 38, "M"),
    (1.7965856976536, 90.562408702746, 30, "M"),
    (1.9335351186162, 91.06932423826, 28, "M"),
    (1.707365374395, 80.02526283857, 20, "M"),
    (1.7709728071584, 78.076259970215, 27, "M"),
    (1.889258520252, 84.519228760393, 30, "M"),
    (1.9805761312032, 95.756977537605, 25, "M"),
    (1.867820220692, 91.046693385985, 29, "M"),
]
df = pd.DataFrame(Data, columns=["height", "weight", "age", "gender"])
df.to_csv("gender_data.csv", index=False)
