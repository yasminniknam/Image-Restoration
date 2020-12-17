# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import shutil
import sys
from subprocess import call
from PIL import Image, ImageFile
from Face_Detection import align_warp_back_multiple_dlib
from Face_Detection import detect_all_dlib

sys.path.insert(0, '/content/photo_restoration/Global')
import test
import detection
sys.path.remove('/content/photo_restoration/Global')

sys.path.insert(0, '/content/photo_restoration/Face_Enhancement')
import test_face
sys.path.remove('/content/photo_restoration/Face_Enhancement')


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="", help="Test images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/home/jingliao/ziyuwan/workspace/codes/PAMI/outputs",
        help="Restored images, please use the absolute path",
    )
    parser.add_argument("--GPU", type=str, default="6,7", help="0,1,2")
    parser.add_argument(
        "--checkpoint_name", type=str, default="Setting_9_epoch_100", help="choose which checkpoint"
    )
    parser.add_argument("--with_scratch", action="store_true")
    opts = parser.parse_args()

    gpu1 = opts.GPU

    # resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    main_environment = os.getcwd()

    
    ## Stage 1: Overall Quality Improve
    # print("Running Stage 1: Overall restoration")
    os.chdir("./Global")
    stage_1_input_dir = opts.input_folder
    stage_1_output_dir = os.path.join(opts.output_folder, "stage_1_restore_output")
    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)

    input_images = []
    input_names = []

    imagelist = os.listdir(stage_1_input_dir)
    imagelist.sort()
    
    for image_name in imagelist:
        input_file = os.path.join(stage_1_input_dir, image_name)
        if not os.path.isfile(input_file):
            print("Skipping non-file %s" % image_name)
            continue
        input_names.append(image_name)
        input_image = Image.open(input_file).convert("RGB")
        input_images.append(input_image)
    

    if not opts.with_scratch:
        input_opts_stage1 = ["--test_mode", "Full", "--Quality_restore", 
                            "--test_input", stage_1_input_dir, "--outputs_dir", stage_1_output_dir, 
                            "--gpu_ids", gpu1]
    
        restored_images = test.test(input_opts_stage1, input_images, input_names)
    
    else:
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")
        
        # input_opts_stage1_command1 = ["--test_path", stage_1_input_dir, "--output_dir", mask_dir,
        #                             "--input_size", "full_size", "--GPU", gpu1]
        
        input_opts_stage1_command1 = ["--test_path", stage_1_input_dir, "--output_dir", mask_dir,
                                    "--input_size", "scale_256", "--GPU", gpu1]

        input_imgs_after_detection, mask_dirs = detection.detection(input_opts_stage1_command1, input_images, input_names)
        
        # input_opts_stage1_command2 = ["--Scratch_and_Quality_restore", "--test_input", new_input,
        #                             "--test_mask", new_mask, "--outputs_dir", stage_1_output_dir,
        #                             "--gpu_ids", gpu1]

        input_opts_stage1_command2 = ["--test_mode", "Scale", "--Scratch_and_Quality_restore", "--test_input", new_input,
                                    "--test_mask", new_mask, "--outputs_dir", stage_1_output_dir,
                                    "--gpu_ids", gpu1]

        restored_images = test.test(input_opts_stage1_command2, input_imgs_after_detection, input_names, mask_loader=mask_dirs)

    ## Solve the case when there is no face in the old photo
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    # for x in os.listdir(stage_1_results):
    #     img_dir = os.path.join(stage_1_results, x)
    #     shutil.copy(img_dir, stage_4_output_dir)

    # print("Finish Stage 1 ...")
    # print("\n")

    ## Stage 2: Face Detection

    # print("Running Stage 2: Face Detection")
    os.chdir(".././Face_Detection")
    stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_2_output_dir = os.path.join(opts.output_folder, "stage_2_detection_output")
    if not os.path.exists(stage_2_output_dir):
        os.makedirs(stage_2_output_dir)
    
    input_opts_stage2 = ["--url", stage_2_input_dir, "--save_url", stage_2_output_dir]
    face_names, faces_detected = detect_all_dlib.detect_all_dlib(input_opts_stage2, restored_images, input_names)
    
    # print("Finish Stage 2 ...")
    # print("\n")

    ## Stage 3: Face Restore
    stage3_names = face_names
    stage3_input = faces_detected
    
    # print("Running Stage 3: Face Enhancement")

    os.chdir(".././Face_Enhancement")
    stage_3_input_mask = "./"
    stage_3_input_face = stage_2_output_dir
    stage_3_output_dir = os.path.join(opts.output_folder, "stage_3_face_output")
    if not os.path.exists(stage_3_output_dir):
        os.makedirs(stage_3_output_dir)
    
        
    input_opts_stage3 = ["--old_face_folder", stage_3_input_face, "--old_face_label_folder", stage_3_input_mask,
                        "--tensorboard_log", "--name", opts.checkpoint_name, "--gpu_ids", gpu1,
                        "--load_size", "256", "--label_nc", "18", "--no_instance", "--preprocess_mode", "resize",
                        "--batchSize", "4", "--results_dir", stage_3_output_dir, "--no_parsing_map"]

    fine_faces = test_face.test_face(input_opts_stage3, stage3_input, stage3_names)
    
    # print("Finish Stage 3 ...")
    # print("\n")

    ## Stage 4: Warp back
    # print("Running Stage 4: Blending")
    os.chdir(".././Face_Detection")
    stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_url", type=str, default="./", help="origin images")
    parser.add_argument("--replace_url", type=str, default="./", help="restored faces")
    parser.add_argument("--save_url", type=str, default="./save")
    input_opts = ["--origin_url", stage_4_input_image_dir, "--replace_url", stage_4_input_face_dir,
                "--save_url", stage_4_output_dir]
    
    opts = parser.parse_args(input_opts)
    align_warp_back_multiple_dlib.align_warp_back_multiple_dlib(opts, restored_images, fine_faces, input_names, face_names)
    # print("Finish Stage 4 ...")
    # print("\n")

    print("All the processing is done. Please check the results.")

