# How to run the Code

First clone down this repo. You may also need to download the BraTS-Reg dataset from this [link](https://www.med.upenn.edu/cbica/brats-reg-challenge/#RegistrationDataRequest7)
Please note that access the dataset needs you to explicitly request it and wait for approval since this dataset was used for BraTS-Reg 2022 challenge. If you just want to test out the effects of our registration methods, you can contact any one of the repo owners. We wil try to send you some sample data to play with.

ITK-C++:
- With/Without mask: Make a build folder the same level as the src folder, then `cmake ../src`, and `make`, to build the executable.
- Once executable built, you can simple type `./RegistrationITK1 inputMovingImageFile inputFixedImageFile segImageFile outputRegisteredMovingImageFile`

Python:
- Without mask: Go into the registration.py file and change the output path to your desired path. Also you may need to install the SimpleITK package. Place the dataset at the same level as the python file for it to properly read inputs. After the paths are set, simply run `python registration.py` in your terminal at the python folder level. Outputs will go to your specified path.
- With mask: If want to test with produced tumor masks, you should not change any of the default image indices since they are the only ones that have masks already generated. Other steps are the same.

Link to presentation slides about this project if you are interested: [Slides](https://docs.google.com/presentation/d/1f_1T9nse-7tONWpjenIg4xz3355Hxe0Z8OnHtK9y_VE/edit?usp=sharing)
