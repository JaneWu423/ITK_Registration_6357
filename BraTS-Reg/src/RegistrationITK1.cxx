#include "itkImage.h"
#include "itkCommand.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkAffineTransform.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkVersorRigid3DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkImageToImageMetricv4.h"
#include "itkImageMaskSpatialObject.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkPoint.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace Eigen;
const unsigned int nDims = 3;
typedef itk::Image <float, nDims> ImageType;


class IterationCallback : public itk::Command {

public:
    typedef IterationCallback Self;
    typedef itk::Command Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    itkNewMacro(IterationCallback);

protected:
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    using OptimizerPointerType = const OptimizerType *;
    
public: 
    void Execute (const itk::Object *caller, const itk::EventObject &event) override {
        OptimizerPointerType optimizer = static_cast <OptimizerPointerType> (caller);
        if (!optimizer) {
            std::cout << "dynamic cast failed, exiting" << std::endl;
            return;
        }
        std::cout << "iteration " << optimizer->GetCurrentIteration() << " " << optimizer->GetValue() << std::endl;
    }

    void Execute ( itk::Object *caller, const itk::EventObject &event) override {
        Execute ( ( const itk::Object * ) caller, event ) ;
    }

};


std::vector<Vector3d> readLandmarks(const std::string& filename) {
    std::vector<Vector3d> landmarks;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return landmarks;
    }
    std::string line;
    std::getline(file, line);
    std::cout << "reading original landmark file." << std::endl;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string value;
        std::vector<std::string> substrings;

        while (std::getline(iss, value, ',')) {
            substrings.push_back(value);
        }

        if (substrings.size() >= 3) {
            double x = std::stod(substrings[1]);
            double y = std::stod(substrings[2]);
            double z = std::stod(substrings[3]);
            landmarks.push_back(Vector3d(x, y, z));
        } else {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
    }

    file.close();
    return landmarks;
}



void writeLandmarks(const std::string& filename, const std::vector<Vector3d>& landmarks) {
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::cout << "Writing transformed landmarks." << std::endl; 
    for (size_t i = 0; i < landmarks.size(); ++i) {
        file << i + 1 << "," << landmarks[i](0) << "," << landmarks[i](1) << "," << landmarks[i](2) << "\n";
    }
    file.close();
}


int main ( int argc, char * argv[] ) {
    if (argc < 6) {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " inputMovingImageFile inputFixedImageFile outputRegisteredMovingImageFile inputCSV outputCSV" << std::endl; 
        return EXIT_FAILURE;
    }

    std::string inputFilename = argv[4];
    std::vector<Vector3d> landmarks = readLandmarks(inputFilename); 
    
    typedef itk::ImageFileReader<ImageType> ImageReaderType;

    /* read moving image */
    ImageReaderType::Pointer movingReader = ImageReaderType::New() ;
    movingReader->SetFileName(argv[1]);
    movingReader->Update();

    /* read fixed image */
    ImageReaderType::Pointer fixedReader = ImageReaderType::New();
    fixedReader->SetFileName(argv[2]);
    fixedReader->Update();

    
    using TransformType = itk::VersorRigid3DTransform<double>;
    using MetricType = itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType>;
    using RegistrationType = itk::ImageRegistrationMethodv4<ImageType, ImageType, TransformType>;
    using FixedLinearInterpolatorType = itk::LinearInterpolateImageFunction<ImageType, double>;
    using MovingLinearInterpolatorType = itk::LinearInterpolateImageFunction<ImageType, double>;
    typedef itk::LinearInterpolateImageFunction <ImageType, double> InterpolatorType;
    typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;

    auto metric = MetricType::New();
    auto registration = RegistrationType::New();
    OptimizerType::Pointer optimizer = OptimizerType::New();
    TransformType::Pointer transformation = TransformType::New();
    FixedLinearInterpolatorType::Pointer fixedInterpolator = FixedLinearInterpolatorType::New();
    MovingLinearInterpolatorType::Pointer movingInterpolator = MovingLinearInterpolatorType::New();

    metric->SetFixedInterpolator(fixedInterpolator);
    metric->SetMovingInterpolator(movingInterpolator);

    registration->SetFixedImage(fixedReader->GetOutput());
    registration->SetMovingImage(movingReader->GetOutput());
    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);

    TransformType::Pointer movingInitialTransform = TransformType::New();
    TransformType::ParametersType initialParameters(movingInitialTransform->GetNumberOfParameters());
    initialParameters[0] = 0.0;  
    initialParameters[1] = 0.0; 
    movingInitialTransform->SetParameters( initialParameters );

    TransformType::Pointer   identityTransform = TransformType::New();
    identityTransform->SetIdentity();

    registration->SetMovingInitialTransform( movingInitialTransform );
    registration->SetFixedInitialTransform( identityTransform );

    constexpr unsigned int numberOfLevels = 1;
    registration->SetNumberOfLevels(numberOfLevels);

    RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize(1);
    shrinkFactorsPerLevel[0] = 1;
    
    RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize(1);
    smoothingSigmasPerLevel[0] = 0;
  
    registration->SetNumberOfLevels ( numberOfLevels );
    registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
    registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
    
    transformation->SetIdentity();
    
    std::cout << "initial relaxation factor: " << optimizer->GetRelaxationFactor() << std::endl;
    optimizer->SetRelaxationFactor(0.75);
    optimizer->SetLearningRate(0.1);
    optimizer->SetMinimumStepLength(0.001);
    optimizer->SetNumberOfIterations(30);
    optimizer->SetReturnBestParametersAndValue(true);
    
    std::cout << "Step length at the beginning of the registration: " << optimizer->GetCurrentStepLength() << std::endl;
    
    IterationCallback::Pointer callback = IterationCallback::New();
    optimizer->AddObserver(itk::IterationEvent(), callback);
    
    registration->Update();
    std::cout << "Optimizer stop condition: " << registration->GetOptimizer()->GetStopConditionDescription() << std::endl;
    std::cout << "Step length at the end of the registration: " << optimizer->GetCurrentStepLength() << std::endl;

    const TransformType::ParametersType finalParameters = registration->GetOutput()->Get()->GetParameters();
    
    const double       versorX = finalParameters[0];
    const double       versorY = finalParameters[1];
    const double       versorZ = finalParameters[2];
    const double       finalTranslationX = finalParameters[3];
    const double       finalTranslationY = finalParameters[4];
    const double       finalTranslationZ = finalParameters[5];
    const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
    const double       bestValue = optimizer->GetValue();
    
    std::cout << std::endl << std::endl;
    std::cout << "Result = " << std::endl;
    std::cout << " versor X      = " << versorX << std::endl;
    std::cout << " versor Y      = " << versorY << std::endl;
    std::cout << " versor Z      = " << versorZ << std::endl;
    std::cout << " Translation X = " << finalTranslationX << std::endl;
    std::cout << " Translation Y = " << finalTranslationY << std::endl;
    std::cout << " Translation Z = " << finalTranslationZ << std::endl;
    std::cout << " Iterations    = " << numberOfIterations << std::endl;
    std::cout << " Metric value  = " << bestValue << std::endl;

      /* write to the new landmark csv file */
    using PointTransformType = itk::VersorRigid3DTransform<double>;
    PointTransformType::Pointer pointTransform = PointTransformType::New();
    
    PointTransformType::VersorType rotationVersor;
    rotationVersor.Set(versorX, versorY, versorZ, 1.0);
    pointTransform->SetRotation(rotationVersor); 
    
    PointTransformType::OutputVectorType translation;
    translation[0] = finalTranslationX;
    translation[1] = finalTranslationY;
    translation[2] = finalTranslationZ;
    pointTransform->SetTranslation(translation);

    std::cout << "Original Landmarks:" << std::endl;
    for (const auto& landmark : landmarks) {
        std::cout << landmark(0) << "," << landmark(1) << "," << landmark(2) << std::endl;
    }

    for (Vector3d& landmark : landmarks) {
        PointTransformType::InputPointType inputPoint;
        inputPoint[0] = landmark(0);
        inputPoint[1] = landmark(1);
        inputPoint[2] = landmark(2);
        PointTransformType::OutputPointType outputPoint = pointTransform->TransformPoint(inputPoint);
        landmark << outputPoint[0], outputPoint[1], outputPoint[2];
    } 
    std::cout << "Transformed Landmarks:" << std::endl;
    for (const auto& landmark : landmarks) {
        std::cout << landmark(0) << "," << landmark(1) << "," << landmark(2) << std::endl;
    }

    std::string outputFilename = argv[5];
    writeLandmarks(outputFilename, landmarks);
    
    typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleFilterType;
    ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

    resampleFilter->SetInput(movingReader->GetOutput());
    resampleFilter->SetReferenceImage(fixedReader->GetOutput()); // where the new resampled image should live?
    resampleFilter->SetUseReferenceImage(true);
    transformation->SetParameters(registration->GetOutput()->Get()->GetParameters());
    resampleFilter->SetTransform(transformation);
    resampleFilter->Update();
    
    typedef itk::ImageFileWriter<ImageType> ImageWriterType;
    ImageWriterType::Pointer myWriter = ImageWriterType::New();
    myWriter->SetFileName(argv[3]); 
    myWriter->SetInput(resampleFilter->GetOutput());
    myWriter->Update();  


    return EXIT_SUCCESS;
}

