
OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_IMAGE_FILE_SUFFIX = ['jpg', '0.jpg', '0.png', 'png', 'jpeg', '0.jpeg', 'webp']
DEFAULT_VIDEO_FILE_SUFFIX = ['mp4', 'video.mp4']

ASPECT_RATIO_1024 = {
    '0.25': [512., 2048.], '0.26': [512., 1984.], '0.27': [512., 1920.], '0.28': [512., 1856.],
    '0.32': [576., 1792.], '0.33': [576., 1728.], '0.35': [576., 1664.], '0.4':  [640., 1600.],
    '0.42':  [640., 1536.], '0.48': [704., 1472.], '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.], '2.09':  [1472.,  704.], '2.4':  [1536.,  640.],
    '2.5':  [1600.,  640.], '2.89':  [1664.,  576.], '3.0':  [1728.,  576.], '3.11':  [1792.,  576.],
    '3.62':  [1856.,  512.], '3.75':  [1920.,  512.], '3.88':  [1984.,  512.], '4.0':  [2048.,  512.],
}


ASPECT_RATIO_1024_norm = {
    '0.68': [832, 1216],
    '0.72': [832, 1152],
    '0.78': [896, 1152], 
    '1.0': [1024, 1024],
    '1.29': [1152,  896], 
    '1.46': [1216,  832],
    '1.75': [1344,  768], 
}

ASPECT_RATIO_512 = {
     '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
     '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
     }


ASPECT_RATIO_512_norm = {
     '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
     '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
     }

ASPECT_RATIO_256 = {
     '0.25': [128.0, 512.0], '0.26': [128.0, 496.0], '0.27': [128.0, 480.0], '0.28': [128.0, 464.0],
     '0.32': [144.0, 448.0], '0.33': [144.0, 432.0], '0.35': [144.0, 416.0], '0.4': [160.0, 400.0],
     '0.42': [160.0, 384.0], '0.48': [176.0, 368.0], '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
     '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
     '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
     '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
     '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
     '1.75': [336.0, 192.0], '2.0': [352.0, 176.0], '2.09': [368.0, 176.0], '2.4': [384.0, 160.0],
     '2.5': [400.0, 160.0], '2.89': [416.0, 144.0], '3.0': [432.0, 144.0], '3.11': [448.0, 144.0],
     '3.62': [464.0, 128.0], '3.75': [480.0, 128.0], '3.88': [496.0, 128.0], '4.0': [512.0, 128.0]
     }