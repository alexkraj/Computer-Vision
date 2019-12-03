# Calculate coordinates of x,y and select 30% in the middle
            horizontal_start, horizontal_end = get_horizontal_center_of_bounding(left, left+width,
                                                                                 get_mean_pixels(top, top + height, left,
                                                                                                 left + width) / 2)
            vertical_start, vertical_end = get_vertical_center_of_bounding(top, top + height,
                                                                           get_mean_pixels(top, top + height, left,
                                                                                           left + width) / 2)
            Z = []
            # Go through all pixels and calculate Z taking into account focal length and baseline
            for x in range(horizontal_start, horizontal_end):
                for y in range(vertical_start, vertical_end):
                    # Calculate Z = (f * 😎 / disparity[y, x];
                    try:
                        if (disparity[y, x] > 0):
                            Z.append((camera_focal_length_px * stereo_camera_baseline_m) / disparity[y, x])
                    except IndexError:
                        continue

            depth = get_median(Z)  # Get depth by calculating a median of 30% of the box pixels in the middle

            drawPred(classIds[i], confidences[i], left, top, left + width, top + height, backup_image,(255, 178, 50),depth)