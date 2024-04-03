%% Load measured scattering matrix (FOV is cropped to reduce data size)
addpath(".\code\");
load("data_LCN.mat");

%% Construct the extended scattering matrix
paramsp = Polarization_params();
paramsp.two_cam([1;-1;0],[1;-1;0],[1;1;0],[1;1;0],[1;-1i],[1;1i]);

% Select the background area on the right!
field.getScatteringMatrix(paramsp); 
%% Tomogram reconstruction
field.coordinates.Nz = 50;   % Number of pixels in the z direction ???
field.coordinates.dz = 2*1.25*constant.wavelength/constant.mediumRI/4; % Size of pixels in the z direction
field.coordinates.update_parameters();

z_shift = 2.34;
use_cuda = true; % Disable this option if you are running out of GPU memory.
potential = -tomogram_unwrap_grad_CUDA(field,z_shift,use_cuda);
%% Visualization

nout = real(potential);
nout = eye(3) - nout /(2*pi*constant.mediumRI/constant.wavelength)^2; 
nout = nout * constant.mediumRI^2;
[P,D] = pageeig(double(nout));
D=sqrt(D).*eye(3);
nout = pagemrdivide(pagemtimes(P,D),P);

Nz = round(field.coordinates.dz/field.coordinates.dx*size(nout,5));

nout1 = imresize5(nout,[size(nout,3) size(nout,3) Nz]);
nout1 = imrotate5(nout1, 2,[0 -1 0],'nearest','crop');
nout1 = imrotate5(nout1, -1,[-1 0 0],'nearest','crop');

[A,D,~]=pagesvd(nout1);

load("colormap2D_inverted.mat")

indz = 64;


x = 30:field.coordinates.Nx;
y = 30:field.coordinates.Nx;

U = flip(squeeze(A(1,1,y,x,indz)),1);
V = flip(-squeeze(A(2,1,y,x,indz)),1);
W = flip(squeeze(A(3,1,y,x,indz)),1);

mask = flip((squeeze(D(1,1,y,x,indz)-D(3,3,y,x,indz))<0.021).*(x>280),1);
subplot(1,2,1)
imagesc(mask*0.8+~mask.*complex2rgb((-1)*exp(1i*angle((U+1i*V).^2)).*abs(angle(abs(W)+1i*sqrt(U.^2+V.^2))/(pi/2)),1,colormap2D));axis image
subplot(1,2,2)
imagesc(flip(squeeze(D(1,1,y,x,indz)-D(3,3,y,x,indz)),1));colormap gray;axis image;clim([0 0.08])
