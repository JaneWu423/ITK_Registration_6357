import SimpleITK as sitk

# Load the brain scan image
# replace with corresponding path
i = 67
image_path = "/Users/janewu/Downloads/2023fall/6357/BraTSReg_Training_Data_v3/BraTSReg_0{}/BraTSReg_0{}_00_0000_t1ce.nii.gz".format(
    i, i
)
image = sitk.ReadImage(image_path)
image = sitk.Cast(image, sitk.sitkFloat32)

# remove background region
threshold_value = 0.0  # Adjust as needed
brain_mask = image > threshold_value

# apply the brain mask to orig image
brain_image = sitk.Mask(image, brain_mask)

# manual selected seed point
seed_point = (91, 128, 105)

# perform region growing using the seed image
tumor_mask = sitk.ConnectedThreshold(
    image,
    seedList=[seed_point],
    lower=1650,
    upper=2400,
)

# combine brain mask with tumor mask
tumor_mask = sitk.BinaryDilate(tumor_mask)
tumor_mask = sitk.Not(tumor_mask)
tumor_mask = sitk.And(tumor_mask, brain_mask)

# write out to specified path
sitk.WriteImage(tumor_mask, "./mask_{}.nii.gz".format(i))
