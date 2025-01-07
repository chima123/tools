#!/usr/bin/env python3
# -*- coding: utf8 -*-

# convert nv12 to bgr image

import sys
import cv2
import os
import numpy as np
import shutil

def nv122bgr(yuv_path, width, height):
    with open(yuv_path, 'rb') as f:
        yuvdata = np.fromfile(f, dtype=np.uint8)
    cv_frmat = cv2.COLOR_YUV2BGR_NV12
    bgr_img = cv2.cvtColor(yuvdata.reshape((height*3//2, width)), cv_frmat)
    return bgr_img


def nv122bgrimg(yuv_path, jpg_path, width, height):
    yuv_path = os.path.abspath(yuv_path)

    try:
        with open(yuv_path, 'rb') as f:
            yuvdata = np.fromfile(f, dtype=np.uint8)
        # cv_frmat = cv2.COLOR_YUV2BGR_NV12
        cv_frmat = cv2.COLOR_YUV2BGR_NV21
        bgr_img = cv2.cvtColor(yuvdata.reshape((height*3//2, width)), cv_frmat)
    except:
        import traceback
        print(f"Failed to convert file: {yuv_path}")
        traceback.print_exc()
        return

    output = "{}/{}.jpg".format(jpg_path, os.path.splitext(os.path.basename(yuv_path))[0])
    print(f"jpgpath: {jpg_path} output file: {output}")
    cv2.imwrite(output, bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def bgr2nv12(image):
    image = image.astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(
        (height * width * 3 // 2, ))
    y = yuv420p[:height * width]
    uv_planar = yuv420p[height * width:].reshape((2, height * width // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((height * width // 2, ))
    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12


def image_from_folder(image_folder_path):
    image_names = []
    if os.path.isdir(image_folder_path):
        ls = os.listdir(image_folder_path)
        for file_name in ls:
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in ['yuv', 'yuyv']:
                image_names.append(file_name)
    return image_names


def convert_image_format(src_folder_path, dst_folder_path, image_width, image_height):
    image_names = image_from_folder(src_folder_path)

    # write
    for i, image_path in enumerate(image_names):
        fpath = os.path.join(src_folder_path, image_path)
        fstat = os.stat(fpath)
        if fstat.st_size > 0:
            try:
                image_BGR = nv122bgr(fpath, image_width, image_height)
            except:
                print(f"Failed to convert file: {fpath}")
                continue

            if False:
                cv2.imshow("image", image_BGR)
                cv2.waitKey(0)
            else:
                cv2.imwrite(os.path.join(dst_folder_path, os.path.splitext(image_path)[
                            0]+".jpg"), image_BGR, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if(i % 100 == 0) and (i > 0):
                print("image num:", i)


if __name__ == "__main__":
    image_width = 5120
    image_height = 960
    # image_width = 1920
    # image_height = 1088
    if len(sys.argv) < 2:
        print("Usage: python3 nv12tobgr.py <path>")

    src_path = os.path.abspath(sys.argv[1])

    if os.path.isfile(src_path):
        dst_path = src_path
        nv122bgrimg(src_path, dst_path, image_width, image_height)
    elif os.path.isdir(src_path):
        dst_path = "jpgs"
        if os.path.exists(dst_path) and os.path.isdir(dst_path):
            try:
                shutil.rmtree(dst_path)
            except OSError as e:
                print(f"Error: {e.strerror}")
        os.makedirs(dst_path)
        dst_path = os.path.abspath(dst_path)
        print(f"dst_psth: {dst_path}")

        for root, dirs, files in os.walk(src_path):
            for file in files:
                if file.endswith(".nv12"):
                    full_path = os.path.join(root, file)
                    print(f"  File: {full_path}")
                    nv122bgrimg(full_path, dst_path, image_width, image_height)


