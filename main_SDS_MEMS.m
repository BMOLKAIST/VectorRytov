%% Load measured fields
addpath(".\code\");
load("data_droplets.mat");
%% Construct the extened scattering matrix 
paramsp = Polarization_params();
paramsp.one_cam([1;0;0],[0;1;0],[1;-1i],[1;1i]); % polarizations of references and illuminations  
field_SDS.getScatteringMatrix(paramsp);
%% Reconstruct the tomogram of an LC droplet in SDS 
field_SDS.coordinates.Nz = 100;   % Number of pixels in the z direction
field_SDS.coordinates.dz = 2*1.25*constant.wavelength/constant.mediumRI/4; % Size of pixels in the z direction
field_SDS.coordinates.update_parameters();
z_shift = 0;


use_cuda = true; % Disable this option if you are running out of GPU memory.
potential = -tomogram_unwrap_grad_CUDA(field_SDS,z_shift,use_cuda);

nout = real(potential);
nout = eye(3) - nout /(2*pi*constant.mediumRI/constant.wavelength)^2; 
nout = nout * constant.mediumRI^2;
[P,D] = pageeig(double(nout));
D=sqrt(D).*eye(3);
nout = pagemrdivide(pagemtimes(P,D),P);

Nz = round(field_SDS.coordinates.dz/field_SDS.coordinates.dx*size(nout,5));
[A,D,~]=pagesvd(nout);
%% top view
load("colormap2D_maxC.mat")
indz = 50;

U = squeeze(A(1,1,:,:,indz));
V = squeeze(A(2,1,:,:,indz));
W = squeeze(A(3,1,:,:,indz));
B = max(0,squeeze(D(1,1,:,:,indz))-constant.mediumRI);

figure(1)
subplot(2,2,1)
imagesc(complex2rgb((-1)*B.*exp(1i*angle((U+1i*V).^2)),0.03,colormap2D));axis image
subplot(2,2,2)
imagesc(squeeze(D(1,1,:,:,indz)-D(3,3,:,:,indz)));colormap gray;axis image;clim([0 0.003])


mask = (B > 0.01).*(squeeze(D(1,1,:,:,indz)-D(3,3,:,:,indz))>0.001);
U = U./sqrt(U.^2+V.^2);
V = V./sqrt(U.^2+V.^2);

for k = 0:9
    ref = 30*[cos(2*pi*k/10),sin(2*pi*k/10)];
    sgn = sign(U.*ref(1)+V.*ref(2));
    U = U.*sgn; V = V.*sgn;
    streamline(U.*mask, V.*mask,169+ref(1),176+ref(2))
    streamline(-U.*mask, -V.*mask,169+ref(1),176+ref(2))
end



%% side view
load("colormap2D_maxC.mat")
ind = 176;

U = squeeze(A(1,1,ind,:,:));
V = squeeze(A(2,1,ind,:,:));
W = squeeze(A(3,1,ind,:,:));
B = max(0,squeeze(D(1,1,ind,:,:))-constant.mediumRI);

figure(1)
subplot(2,2,3)
imagesc(complex2rgb((-1)*imresize(B.'.*exp(1i*angle((U.'+1i*W.').^2)),[Nz size(nout,3) ]),0.03,colormap2D));axis image
subplot(2,2,4)
imagesc(imresize(squeeze(D(1,1,ind,:,:)-D(3,3,ind,:,:)).',[Nz size(nout,3) ]));colormap gray;axis image;clim([0 0.003])

mask = (imresize(B.',[Nz size(nout,3)])>0.02);
mask(110:121,164:172)=0;
U=imresize(U.',[Nz size(nout,3) ]);
W=imresize(W.',[Nz size(nout,3) ]);
U = U./sqrt(U.^2+W.^2);
W = W./sqrt(U.^2+W.^2);
for k = 0:9
    ref = 25*[cos(2*pi*k/10),sin(2*pi*k/10)];
    sgn = sign(U.*ref(1)+W.*ref(2));
    U = U.*sgn; W = W.*sgn;
    streamline(U.*mask, W.*mask,166+ref(1),116+ref(2))
    streamline(-U.*mask, -W.*mask,166+ref(1),116+ref(2))
end


