%% Load measured fields
addpath(".\code\");
load("data_droplets.mat");

%% Construct the extened scattering matrix 
paramsp = Polarization_params();
paramsp.one_cam([1;0;0],[0;1;0],[1;-1i],[1;1i]); % polarizations of references and illuminations  
field_PVA.getScatteringMatrix(paramsp);

%% Reconstruct the tomogram of an LC droplet in PVA
field_PVA.coordinates.Nz = 100;   % Number of pixels in the z direction
field_PVA.coordinates.dz = 2*1.25*constant.wavelength/constant.mediumRI/4; % Size of pixels in the z direction
field_PVA.coordinates.update_parameters();
z_shift = 0;

use_cuda = true; % Disable this option if you are running out of GPU memory.
potential = -tomogram_unwrap_grad_CUDA(field_PVA,z_shift,use_cuda);
%% top view
load("colormap2D_maxC.mat")

nout = real(potential);
nout = eye(3) - nout /(2*pi*constant.mediumRI/constant.wavelength)^2; 
nout = nout * constant.mediumRI^2;
[P,D] = pageeig(double(nout));
D=sqrt(D).*eye(3);
nout = pagemrdivide(pagemtimes(P,D),P);


[A,D,~]=pagesvd(nout);

ind = 51;

U = squeeze(A(1,1,:,:,ind));
V = squeeze(A(2,1,:,:,ind));
W = squeeze(A(3,1,:,:,ind));
B = max(0,squeeze(D(1,1,:,:,ind))-constant.mediumRI);

figure(1)
subplot(2,2,1)
imagesc(complex2rgb((-1)*B.*exp(1i*angle((U+1i*V).^2)),0.03,colormap2D));axis image
subplot(2,2,2)
imagesc(squeeze(D(1,1,:,:,ind)-D(3,3,:,:,ind)));colormap gray;axis image;clim([0 0.003])


mask = (B > 0.01).*(squeeze(D(1,1,:,:,ind)-D(3,3,:,:,ind))>0.0008);
U = U./sqrt(U.^2+V.^2);
V = V./sqrt(U.^2+V.^2);
ref = [114 35+20];
sgn = sign(U.*ref(1)+V.*ref(2));
U = U.*sgn; V = V.*sgn;

pts = linspace(120,235,8);
streamline(U.*mask, V.*mask,180.*ones(size(pts)),pts)
streamline(-U.*mask, -V.*mask,180.*ones(size(pts)),pts)


%% side view
load("colormap2D_maxC.mat")

nout = rotateTensorField(real(potential),-25);
nout = eye(3) - nout /(2*pi*constant.mediumRI/constant.wavelength)^2; 
nout = nout * constant.mediumRI^2;
[P,D] = pageeig(double(nout));
D=sqrt(D).*eye(3);
nout = pagemrdivide(pagemtimes(P,D),P);

Nz = round(field_PVA.coordinates.dz/field_PVA.coordinates.dx*size(nout,5));
[A,D,~]=pagesvd(nout);

ind = 164;

U = squeeze(A(1,1,ind,:,:));
V = squeeze(A(2,1,ind,:,:));
W = squeeze(A(3,1,ind,:,:));
B = max(0,squeeze(D(1,1,ind,:,:))-constant.mediumRI);

figure(1)
subplot(2,2,3)
imagesc(complex2rgb((-1)*imresize(B.'.*exp(1i*angle((U.'+1i*W.').^2)),[Nz size(nout,3) ]),0.03,colormap2D));axis image
subplot(2,2,4)
imagesc(imresize(squeeze(D(1,1,ind,:,:)-D(3,3,ind,:,:)).',[Nz size(nout,3) ]));colormap gray;axis image;clim([0 0.003])

mask = (imresize(B.',[Nz size(nout,3) ])>0.01).*(imresize(squeeze(D(1,1,ind,:,:)-D(3,3,ind,:,:)).',[Nz size(nout,3) ])>0.0008);
U=imresize(U.',[Nz size(nout,3) ]);
W=imresize(W.',[Nz size(nout,3) ]);
U = U./sqrt(U.^2+W.^2);
W = W./sqrt(U.^2+W.^2);
pts = linspace(60,170,8);
streamline(U.*sign(U).*mask, W.*sign(U).*mask,180.*ones(size(pts)),pts)
streamline(-U.*sign(U).*mask, -W.*sign(U).*mask,180.*ones(size(pts)),pts)


subplot(2,2,1)
hold on
plot([1 size(nout,3)],[size(nout,3)/2-(size(nout,3)/2-ind)*acos(25*pi/180)-size(nout,3)/2*tan(25*pi/180), size(nout,3)/2-(size(nout,3)/2-ind)*acos(25*pi/180)+size(nout,3)/2*tan(25*pi/180)],'-w');
hold off


