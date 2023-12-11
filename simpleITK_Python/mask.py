import SimpleITK as sitk

# Load the brain scan image
i = 67
image_path = "/Users/janewu/Downloads/2023fall/6357/BraTSReg_Training_Data_v3/BraTSReg_0{}/BraTSReg_0{}_00_0000_t1ce.nii.gz".format(
    i, i
)
image = sitk.ReadImage(image_path)
image = sitk.Cast(image, sitk.sitkFloat32)

# Threshold to identify the brain region
threshold_value = 0.0  # Adjust as needed
brain_mask = image > threshold_value

# Apply the brain mask to the original image
brain_image = sitk.Mask(image, brain_mask)

# seed_image = sitk.Image(brain_image.GetSize(), sitk.sitkFloat32)
# seed_image.CopyInformation(brain_image)

seed_point = (91, 128, 105)  # Replace with the coordinates of your seed point
# seed_point1 = (130, 85, 100)  # Replace with the coordinates of your seed point
# seed_image[seed_point] = 1


# segmented_image = sitk.ShapeDetectionLevelSet(
#     image,
#     seed_image,
#     numberOfIterations=100,
#     propagationScaling=1.0,
#     curvatureScaling=1.0,
# )


# seed_image[seed_point] = 1  # Set the seed point in the seed image

# Perform region growing using the seed image
tumor_mask = sitk.ConnectedThreshold(
    image,
    seedList=[seed_point],
    lower=1650,
    upper=2400,
    # seedContribution=1.0,
)

tumor_mask = sitk.BinaryDilate(tumor_mask)
tumor_mask = sitk.Not(tumor_mask)
tumor_mask = sitk.And(tumor_mask, brain_mask)

sitk.WriteImage(tumor_mask, "./mask_{}.nii.gz".format(i))

# # Adaptive Thresholding to create binary mask
# tumor_mask = sitk.BinaryThreshold(
#     brain_scan,
#     lowerThreshold=0,
#     upperThreshold=sitk.OtsuThreshold(brain_scan),
#     insideValue=1,
#     outsideValue=0,
# )

# # Connected Component Analysis (Optional)
# tumor_mask_connected = sitk.ConnectedComponent(tumor_mask)
# tumor_mask_largest = sitk.RelabelComponent(
#     tumor_mask_connected, 1
# )  # Keep only the largest connected component

# # Dilation (Optional)
# tumor_mask_dilated = sitk.BinaryDilate(
#     tumor_mask_largest, radius=3
# )  # Adjust the radius as needed

# # Save the mask (Optional)
# sitk.WriteImage(tumor_mask_dilated, "./mask.nii.gz")
