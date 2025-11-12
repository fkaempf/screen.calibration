% calibrate_fisheye_KD_from_path.m
% Produces K (3x3 for undistorted pinhole) and D (MATLAB fisheye coeffs).

clear; clc;

%% INPUTS
calib_dir    = 'D:\screen.calibration\checkerboard_samples';
boardSize    = [7 10];     % inner corners [rows cols]
squareSizeMM = 25;         % edge length of one square in millimetres

%% Collect images
exts = {'*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp'};
files = [];
for k = 1:numel(exts)
    files = [files; dir(fullfile(calib_dir, exts{k}))]; %#ok<AGROW>
end
assert(~isempty(files), 'No calibration images found.');
imageFileNames = fullfile({files.folder},{files.name})';

%% Detect checkerboard corners
[imagePoints, detectedBoardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
assert(all(detectedBoardSize==boardSize), ...
    'Detected board %s != expected %s', mat2str(detectedBoardSize), mat2str(boardSize));

imageFileNames = imageFileNames(imagesUsed);
I0 = imread(imageFileNames{71});
imageSize = [size(I0,1) size(I0,2)];   % [rows cols]
worldPoints = generateCheckerboardPoints(boardSize, squareSizeMM);

%% Calibrate fisheye (Scaramuzza/Taylor model)
[fisheyeParams, imagesUsed2, estimationErrors] = ...
    estimateFisheyeParameters(imagePoints, worldPoints, imageSize); %#ok<ASGLU>

%% Export K and D
probe = [0 0; imageSize(2) 0; 0 imageSize(1); imageSize(2) imageSize(1)];
[~, camIntrinsics] = undistortFisheyePoints(probe, fisheyeParams.Intrinsics);
K = camIntrinsics.IntrinsicMatrix.';                          % 3x3 pinhole K (for UNDISTORTED images)
D = fisheyeParams.Intrinsics.MappingCoefficients;             % [a0 a2 a3 a4]
C = fisheyeParams.Intrinsics.DistortionCenter;                % [cx cy] pixels
Stretch = fisheyeParams.Intrinsics.StretchMatrix;             % 2x2

disp('K ='); disp(K);
disp('D = [a0 a2 a3 a4]'); disp(D);
disp('Distortion center [cx cy] ='); disp(C);
disp('Stretch matrix ='); disp(Stretch);

%% Save
save(fullfile(calib_dir,'fisheye_calib_results.mat'), ...
     'fisheyeParams','estimationErrors','K','D','C','Stretch', ...
     'boardSize','squareSizeMM','imageSize','imageFileNames');

% % Visual check (optional)
J = undistortFisheyeImage(I0, fisheyeParams.Intrinsics, 'ScaleFactor', 0.2);;
figure; imshowpair(I0, J, 'montage'); title('Original | Undistorted');

K = camIntrinsics.IntrinsicMatrix.';
a = fisheyeParams.Intrinsics.MappingCoefficients; % [a0 a2 a3 a4]
D = [a(2) a(3) a(4) 0];                           % OpenCV expects [k1 k2 k3 k4]
save("D:\screen.calibration\fisheye_to_opencv.mat",'K','D');
