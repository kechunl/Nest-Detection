class Config:
    slide_window = 1000
    overlap_scale = 0.5

    resize_factor = 0.5

    num_classes = 2  # foreground + 1 background
    color_map = {int('005500', 16): 1,
                 int('ffffff', 16): 1,
                 int('00ff00', 16): 1,
                 int('aa00ff', 16): 1}  # INSI, JN, DM, DN
    # color_map = {int('005500', 16): 1,
    #              int('ffffff', 16): 1}  # INSI, JN
    # color_map = {int('00ff00', 16): 1,
    #              int('aa00ff', 16): 1}  # DM, DN

    # num_classes = 3  # foreground + 1 background
    # color_map = {int('005500', 16): 1,
    #              int('ffffff', 16): 1,
    #              int('00ff00', 16): 2,
    #              int('aa00ff', 16): 2}  # INSI, JN, DM, DN

    # color_map = {int('005500', 16): 1,
    #              int('ffffff', 16): 1,
    #              int('00ff00', 16): 1,
    #              int('aa00ff', 16): 1,
    #              int('ff0000', 16): 2,
    #              int('ff5500', 16): 2,
    #              int('55ffff', 16): 2,
    #              int('550000', 16): 2}  # INSI, JN, DM, DN, BV, INF, HF, ED

    original_img_dir = '/projects/patho1/Kechun/NestDetection/dataset_corrected/ROI/split'
    mask_dir = '/projects/patho1/Kechun/NestDetection/dataset_corrected/ROI/masks_plus_bg'
    # patch_dir = '/projects/patho1/Kechun/NestDetection/dataset/patch/patch_{}_{}_{}_{}_{}'.format("EP", slide_window, overlap_scale, resize_factor, num_classes)
    # resize_dir = '/projects/patho1/Kechun/NestDetection/dataset/full_image/ROI_{}_{}'.format("EP", resize_factor)
    patch_dir = '/projects/patho1/Kechun/NestDetection/dataset_corrected/patch/patch_{}_{}_{}_{}'.format(slide_window, overlap_scale, resize_factor, num_classes)
    resize_dir = '/projects/patho1/Kechun/NestDetection/dataset_corrected/full_image/ROI_{}_{}'.format('cat_{}'.format(num_classes), resize_factor)


dataset_cfg = Config()