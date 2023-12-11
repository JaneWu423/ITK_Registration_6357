# Load necessary modules
import SimpleITK as sitk
import csv
import numpy as np
import os


# find all csv in a folder
def list_csv_files_recursive(folder_path):
    csv_files = []
    for foldername, subfolder, filenames in os.walk(folder_path):
        temp_csv = []
        for filename in filenames:
            if filename.endswith(".csv"):
                temp_csv.append(os.path.join(foldername, filename))
        if len(temp_csv) == 2:
            temp_csv.sort()
            csv_files.append(temp_csv)
    csv_files.sort()
    return csv_files


# find all img in a folder
def list_img_files_recursive(folder_path):
    img_files = []
    for foldername, subfolder, filenames in os.walk(folder_path):
        temp_img = []
        for filename in filenames:
            # find the t1ce images only
            if filename.endswith("t1ce.nii.gz"):
                temp_img.append(os.path.join(foldername, filename))
        if len(temp_img) == 2:
            temp_img.sort()
            img_files.append(temp_img)
    img_files.sort()

    return img_files


# run the demons registration without using ImageRegistrationMethod
# this is the same as the other run_register function
# but this one is using older demons registration method calls


# problem: can not apply tumor mask...
def run_demon(imgs, csvs, masks, lst):
    cnt = 0
    for i in lst:
        i = i - 1

        def command_iteration(filter):
            print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")

        fixed = sitk.ReadImage(imgs[i][1], sitk.sitkFloat32)

        moving = sitk.ReadImage(imgs[i][0], sitk.sitkFloat32)

        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        moving = matcher.Execute(moving, fixed)

        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(50)

        # std dev for Gaussian smoothing of displacement field
        demons.SetStandardDeviations(1.0)

        demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

        displacementField = demons.Execute(fixed, moving)

        print("-------")
        print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
        print(f" RMS: {demons.GetRMSChange()}")

        transform = sitk.DisplacementFieldTransform(displacementField)
        input_csv_path = csvs[i][0]

        landmark_points = []
        # read in csv file and store the points in a list
        with open(input_csv_path, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            landmark_points = [
                [float(row[1]), float(row[2]), float(row[3])] for row in csvreader
            ]
        for pts in landmark_points:
            print("Base Landmarks", pts)

        # Apply the transformation to each point
        transformed_landmark_points_sitk = [
            transform.TransformPoint(point) for point in landmark_points
        ]
        for pts in transformed_landmark_points_sitk:
            print("Transformed Landmarks", pts)

        # Convert the transformed points back to a NumPy array
        transformed_landmark_points_arr = [
            [point[0], point[1], point[2]] for point in transformed_landmark_points_sitk
        ]

        # write out transform landmarks to csv
        output_csv_path = "/Users/janewu/Downloads/2023fall/6357/BraTSReg_Training_Data_v3/csv1/transformed_landmarks_{}.csv".format(
            i + 1
        )
        header = ["Landmark", "X", "Y", "Z"]

        with open(output_csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)

            for j, point in enumerate(transformed_landmark_points_arr, start=1):
                csvwriter.writerow([j, point[0], point[1], point[2]])

        print(f"Transformed landmark points saved to {output_csv_path}")
        cnt += 1


# same as above demons registration but using ImageRegistrationMethod
# also able to accept a mask image to be used when doing metric calculation
def run_register(imgs, csvs, masks, lst, output_path):
    has_mask = masks is not None
    cnt = 0
    for i in lst:
        i = i - 1
        print("Start {}th registration".format(i + 1))
        moving_image_path = imgs[i][0]

        fixed_image_path = imgs[i][1]

        fixed_image = sitk.ReadImage(fixed_image_path)
        moving_image = sitk.ReadImage(moving_image_path)

        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat64)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat64)

        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        moving_image = matcher.Execute(moving_image, fixed_image)

        size = fixed_image.GetSize()

        # Create a displacement field with the same size as the fixed image
        displacement_size = list(size)
        displacement = sitk.Image(displacement_size, sitk.sitkVectorFloat64)
        displacement.SetSpacing(fixed_image.GetSpacing())
        displacement.SetOrigin(fixed_image.GetOrigin())
        displacement.SetDirection(fixed_image.GetDirection())
        # Set up the displacement field transform
        displacement_transform = sitk.DisplacementFieldTransform(displacement)

        # Set up the Demons registration
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsDemons()
        if has_mask:
            registration.SetMetricMovingMask(masks[cnt])  # if you have a mask
        registration.SetInitialTransform(displacement_transform)
        # registration.SetGaussianSmoothingVarianceForTheUpdateField(1.0)
        registration.SetOptimizerAsGradientDescent(
            learningRate=0.1,
            numberOfIterations=50,
        )

        registration.SetInitialTransform(displacement_transform)

        print("Before Execute Registration")

        transform = registration.Execute(fixed_image, moving_image)
        print("After Execute Registration")

        print("Before Resample")
        registered_image = sitk.Resample(
            moving_image,
            fixed_image,
            transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
        print("After Resample")

        input_csv_path = csvs[i][0]

        landmark_points = []
        with open(input_csv_path, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)  # Skip the header
            landmark_points = [
                [float(row[1]), float(row[2]), float(row[3])] for row in csvreader
            ]
        print("Base Landmarks")
        for pts in landmark_points:
            print(pts)

        # Apply the transformation to each point
        transformed_landmark_points_sitk = [
            transform.TransformPoint(point) for point in landmark_points
        ]
        print("Transformed Landmarks")
        for pts in transformed_landmark_points_sitk:
            print(pts)

        # Convert the transformed points back to a NumPy array
        transformed_landmark_points_arr = [
            [point[0], point[1], point[2]] for point in transformed_landmark_points_sitk
        ]

        # write out transformed landmarks to csv
        output_csv_path = output_path + "csv/transformed_landmarks_{}.csv".format(i + 1)
        header = ["Landmark", "X", "Y", "Z"]

        with open(output_csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)

            for j, point in enumerate(transformed_landmark_points_arr, start=1):
                csvwriter.writerow([j, point[0], point[1], point[2]])

        print(f"Transformed landmark points saved to {output_csv_path}")

        # Save the registered image to a file
        registered_image_path = (
            output_path + "img_results/registered_image_{}.nii.gz".format(i + 1)
        )
        sitk.WriteImage(registered_image, registered_image_path)
        print(f"Registered image saved to {registered_image_path}")

        cnt += 1


if __name__ == "__main__":
    # find all csvs and imgs files in our dataset folder
    dataset_path = "./BratsReg_Training_Data_v3"
    output_path = "/Users/janewu/Downloads/2023fall/6357/BraTSReg_Training_Data_v3/"
    csvs = list_csv_files_recursive(dataset_path)
    imgs = list_img_files_recursive(dataset_path)

    # specify interested images' index
    # default are those ten which has masks produced
    lst = [2, 4, 5, 10, 12, 16, 22, 24, 29, 67]
    masks = None

    #####----- comment following out if you want to run without masks
    mask_names = ["./masks/mask_{}.nii.gz".format(i) for i in lst]
    masks = [sitk.ReadImage(names) for names in mask_names]

    # run demons registration
    run_register(imgs, csvs, masks, lst, output_path)
