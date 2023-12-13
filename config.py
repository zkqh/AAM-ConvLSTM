#####################################################################################
# Data Loader parameters
#####################################################################################
# Sequence length
sequence_length = 10
# Image resolution (height, width)
resolution = (240, 320)
# Path to the folder containing the RGB frames
#frames_dir = '/media/yf302/Lenovo/DHP-TensorFlow-results0.0/frames'
frames_dir = '/media/yf302/Lenovo/vr-eyetracking/frames'
#frames_dir='E:\\vr-eyetracking\\VR\\VREYEDATASET\\videos\\videos'
# Path to the folder containing the optical flow
optical_flow_dir = 'data/optical_flow'
#gt_dir = '/media/yf302/Lenovo/DHP-TensorFlow-results0.0/groundtruth_heatmaps'
#fixation_dir='/media/yf302/Lenovo/DHP-TensorFlow-results0.0/groundtruth_scanpaths/'
# Path to the folder containing the ground truth saliency maps
gt_dir = '/media/yf302/Lenovo/vr-eyetracking/saliency'
fixation_dir='/media/yf302/Lenovo/vr-eyetracking/fixation'
# Txt file containing the list of video names to be used for training
videos_train_file = '/media/vilab/11be89dd-c494-4cd1-a2b7-cb0fd483b60a/vilab/saliency_results/全景视频显著性检测软件/data/train_split_VRET.txt'
# Txt file containing the list of video names to be used for testing
videos_val_file = '/media/vilab/11be89dd-c494-4cd1-a2b7-cb0fd483b60a/vilab/saliency_results/全景视频显著性检测软件/data/test_split_VRET.txt'

#####################################################################################
# Training parameters
#####################################################################################
# Batch size
batch_size = 1
# Nº of epochs
epochs = 240
# Learning rate
lr = 0.00003
#lr = 0.8
# Hidden dimension of the model (SST-Sal uses hidden_dim=36)
hidden_dim =9
# Percentage of training data intended to validation
validation_split = 0.2
# Name of the model ( for saving pruposes)
model_name = 'sst-attention_aba'
# Path to the folder where the checkpoints will be saved
ckp_dir = '/media/yf302/Lenovo/0807sst/checkpoints'
# Path to the folder where the model will be savedp
models_dir ='/media/yf302/Lenovo/0807sst/models'
# Path to the folder where the training logs will be saved
runs_data_dir = 'runs'

#####################################################################################
# Inference parameters
#####################################################################################
# Path to the folder containing the model to be used for inferenCE
#inference_model ='/home/yf302/Desktop/teacher_wan/0807sst/models/SST-vr_eyetracking_20231123-170740/cc0.2274_model.pth'
#inference_model ='/home/yf302/Desktop/teacher_wan/0807sst/models/SST-Sal_20231121-140349/cc0.2714_model.pth'
#inference_model ='/home/yf302/Desktop/teacher_wan/0807sst/models/SST-Sal_20231117-162320/cc0.7707_model.pth'
inference_model ='/media/yf302/Lenovo/0807sst/models/SST-Sal_20231110-144331/cc0.6557_model.pth'
#inference_model ='/home/yf302/Desktop/teacher_wan/0807sst/models/SST-Sal_20231117-112209/cc0.773_model.pth'
#inference_model = '/home/yf302/Desktop/teacher_wan/0807sst/models/SST-Sal_20230819-213259/cc0.6554_model.pth'
#inference_model ='/home/yf302/Desktop/teacher_wan/0807sst/models/SST-Sal_20231105-201331/cc0.5437model.pth'
#inference_model ='/home/yf302/Desktop/teacher_wan/0807sst/models/SST-Sal_20230904-162059/cc0.6541_model.pth'
# inference_model='/home/yf302/Desktop/teacher_wan/0807sst/models/SST-Sal_20231106-161445/cc0.4244_model.pth'
#inference_model ='/home/yf302/Desktop/teacher_wan/0807sst/SST_Sal_wo_OF.pth'
# Path to the folder where the inference results will be saved
# results_dir = '/media/vilab/11be89dd-c494-4cd1-a2b7-cb0fd483b60a/vilab/全景视频显著性测试软件-分析结果/显著性图'
results_dir = '/media/yf302/Lenovo/our_model_result_dir'
# Path to the folder containing the videos to be used for inference
#videos_folder = 'data/videos'
videos_folder='/media/vilab/141E39B41E39902A/VR_sal/videos'
# Indicates if the model used for inference is trained with or without optical flow
of_available = False

train_set=[
    'A380',
    'AcerEngine',
    'AcerPredator',
    'AirShow',
    'BFG',
    'Bicycle',
    'Camping',
    'CandyCarnival',
    'Castle',
    'Catwalks',
    'CMLauncher',
    'CS',
    'DanceInTurn',
    'DrivingInAlps',
    'Egypt',
    'F5Fighter',
    'Flight',
    'GalaxyOnFire',
    'Graffiti',
    'GTA',
    'HondaF1',
    'IRobot',
    'KasabianLive',
    'Lion',
    'LoopUniverse',
    'Manhattan',
    'MC',
    'MercedesBenz',
    'Motorbike',
    'Murder',
    'Orion',
    'Parachuting',
    'Parasailing',
    'Pearl',
    'Predator',
    'ProjectSoul',
    'Rally',
    'RingMan',
    'Roma',
    'Shark',
    'Skiing',
    'Snowfield',
    'SnowRopeway',
    'Square',
    'StarWars',
    'StarWars2',
    'Stratosphere',
    'StreetFighter',
    'Supercar',
    'SuperMario64',
    'Surfing',
    'SurfingArctic',
    'TalkingInCar',
    'Terminator',
    'TheInvisible',
    'Village',
    'VRBasketball',
    'Waterskiing',
    'WesternSichuan',
    'Yacht',
]

test_set = [
    'WaitingForLove',
    'SpaceWar',
    'KingKong',
    'SpaceWar2',
    'Guitar',
    'BTSRun',
    'CMLauncher2',
    'Symphony',
    'RioOlympics',
    'Dancing',
    'StarryPolar',
    'InsideCar',
    'Sunset',
    'Waterfall',
    'BlueWorld'
]

VR_TRAIN_set=['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022',  '035', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057',  '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '085', '087', '088', '089', '090', '091', '092', '093', '094', '095', '109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195',
         '196','197','198','199','200','201','202']

VR_Test_set=['023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034','058', '059', '060', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071','096', '097', '098', '099', '100', '101', '102', '103', '104','105','106','131','132','133','134','135','136','137','138','139','140','141','142','143','166','167','168','169','170','171','172','173','174','175','176','177','178','203','204','205','206','208']

