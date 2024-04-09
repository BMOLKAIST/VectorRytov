function V = tomogram_unwrap_grad_CUDA(field,z_shift,use_cuda)

iter = size(field.field_tensor,5);
coordinates = field.coordinates;
constant = field.constant;
V = complex(gpuArray(single(zeros(3,3,coordinates.Nx,coordinates.Nx,coordinates.Nz))));
Vp = complex(gpuArray(single(zeros(3,3,coordinates.Nx,coordinates.Nx,coordinates.Nz))));
V0 = gpuArray(single(zeros(3,3,coordinates.Nx,coordinates.Nx,iter)));
A = gpuArray(single(zeros(3,3,coordinates.Nx,coordinates.Nx,iter)));
B = gpuArray(single(zeros(3,3,1,1,iter)));
Uz = gpuArray(single(zeros(1,1,coordinates.Nx,coordinates.Nx,iter)));
if ~use_cuda
    mask = gpuArray(single(zeros(1,1,coordinates.Nx,coordinates.Nx,iter)));
end
ref = [];


permutation = zeros(3,3,6);
perm_idx = perms([1,2,3]);
eye3 = eye(3);

for ind = 1:6
    permutation(:,:,ind) = eye3(perm_idx(ind,:),:);
end

for idx = 1:iter
    fprintf("Unwrapping... %d completed out of %d\n",idx,iter)
    uin = field.u_in(:,idx);
    field_tensor = field.field_tensor(:,:,:,:,idx);
    Q = single(coordinates.Qz(uin,constant));
    field_tensor = ifft(ifft(fft(fft(field_tensor,[],3),[],4).*exp(1i*2*pi*Q*z_shift),[],3),[],4);
    Q = gpuArray(Q);

    [P,D] = pageeig(field_tensor);
    D=(log(abs(D)+1e-8)+1i*(angle(D))).*eye(3);
    P = gpuArray(single(P));D = gpuArray(single(D));
    
    % sorting and unwrapping
    [~,dim_uin] = max(pagemtimes(uin' , P),[],2);
    S = [0 0 1;0 1 0;1 0 0].*(dim_uin == 1) + [1 0 0;0 0 1;0 1 0].*(dim_uin == 2) + eye(3).*(dim_uin == 3);
    S = pagemtimes(S,eye(3) + ([0 1 0;1 0 0;0 0 1]-eye(3)).*(unwrap_phase(pagemtimes(P,S),pagemtimes(pagemtimes(pagetranspose(S),D),S))<0));
    Ds = pagemtimes(pagemtimes(pagetranspose(S),D),S);


    for ind = 1:3
        if ind ~= 3      
            Ds(ind,ind,:,:) = squeeze(real(Ds(ind,ind,:,:))) + 1i*(unwrap2_Lp(squeeze(imag(Ds(ind,ind,:,:))),0,ref));
        else
            Ds(ind,ind,:,:) = 0;
        end
    end


    % correct global phases
    if idx == 1
        figure;imagesc(squeeze(imag(Ds(1,1,:,:)))); axis image; colormap gray; colorbar
        title('Select a background area')
        r = drawrectangle;
        bg = r.Position;
        bg = [round(bg(2)), round(bg(2))+round(bg(4))-1, round(bg(1)), round(bg(1))+round(bg(3))-1 ];
    end
    Ds = Ds - 1i * 2*pi*round(mean(imag(Ds(:,:,bg(1):bg(2),bg(3):bg(4))),[3,4])/(2*pi));


    
    D = pagemtimes(pagemtimes(S,Ds),pagetranspose(S));

    % Variables to be mapped onto Ewald spheres

    field_tensor = pagemrdivide(pagemtimes(P,D),P);

    
    NAmask = circshift(coordinates.NAmask(constant),[0 0 -round(uin(2)/coordinates.dux) -round(uin(1)/coordinates.dux)]);

    field_tensor = fft(fft(field_tensor,coordinates.Nx,3),coordinates.Nx,4);
    field_tensor = 1./(1i/(4*pi)./(Q+uin(3))).*field_tensor.*NAmask/coordinates.dz;

    V0(:,:,:,:,idx) = field_tensor;
    A(:,:,:,:,idx) = Green_Dyadic(coordinates, constant, uin);
    B(:,:,1,1,idx) = (eye(3)-uin*transpose(uin)/(transpose(uin)*uin));
    Uz(1,1,:,:,idx) = Q;
    if ~use_cuda
        mask(1,1,:,:,idx) = NAmask;
    end
end

%% parameters for gradient descent and CUDA
step = 0.8/coordinates.Nz/iter;%
step2 = 4/iter;%4
num_epoch = 100;

z = gpuArray(single(coordinates.z));

size_2D = coordinates.Nx*coordinates.Nx;
size_z = coordinates.Nz;
size_angle = iter;
res_z=gather(single(z(2)-z(1)));
start_z=gather(single(z(1)));

kern = parallel.gpu.CUDAKernel("DTT_fast_kernel.ptx","DTT_fast_kernel.cu");
kern.ThreadBlockSize = [size_z, 1, 1];
kern.GridSize = [size_2D, 1, 1];

used_shared_memory = 0;
real_sz = 1;
complex_sz = 2;

pos_A_shared = used_shared_memory;
used_shared_memory = used_shared_memory + 9 * size_angle * real_sz;
pos_B_shared = used_shared_memory;
used_shared_memory = used_shared_memory + 9 * size_angle * real_sz;
pos_Uz_shared = used_shared_memory;
used_shared_memory = used_shared_memory + 1 * size_angle * real_sz;
if mod(used_shared_memory,2)~=0; used_shared_memory=used_shared_memory+1; end;
pos_V0_shared = used_shared_memory;
used_shared_memory = used_shared_memory + 9 * size_angle * complex_sz;
if mod(used_shared_memory,2)~=0; used_shared_memory=used_shared_memory+1; end;
pos_Vout_shared = used_shared_memory;
used_shared_memory = used_shared_memory + 9 * size_z * complex_sz;
if mod(used_shared_memory,2)~=0; used_shared_memory=used_shared_memory+1; end;
pos_Vp_shared = used_shared_memory;
used_shared_memory = used_shared_memory + 9 * size_z * complex_sz;
if mod(used_shared_memory,2)~=0; used_shared_memory=used_shared_memory+1; end;
pos_temp_shared = used_shared_memory;
used_shared_memory = used_shared_memory + size_z * complex_sz;
pos_tmat_shared = used_shared_memory;
used_shared_memory = used_shared_memory + 9 * complex_sz;

kern.SharedMemorySize=used_shared_memory*4;

%% Map fields onto Ewald spheres using the gradient descent method

for epoch = 1:num_epoch
    if ~use_cuda
        G = zeros(3,3,coordinates.Nx,coordinates.Nx,coordinates.Nz,'single','gpuArray')+1i;
        for idx = 1:iter
            exp_factor=exp(1i*2*pi*Uz(1,1,:,:,idx).*z);
            G = G + mask(1,1,:,:,idx).*exp_factor.*pagemtimes(pagemtimes(A(:,:,:,:,idx),sum(conj(exp_factor).*Vp,5)-V0(:,:,:,:,idx)),B(:,:,:,:,idx));
        end
        G = (G + pagetranspose(G))/2;
    else
        G=Vp;
        G=feval(kern,A, B, Uz, V0, G,...
            pos_A_shared,pos_B_shared,pos_Uz_shared,pos_V0_shared,pos_Vout_shared,pos_Vp_shared,pos_temp_shared,pos_tmat_shared,size_2D,size_z,size_angle,...
            res_z,start_z);
    end
    
    Vp = (1-step2)*Vp - step * G +(eye(3).*step2/3).*(Vp(1,1,:,:,:)+Vp(2,2,:,:,:)+Vp(3,3,:,:,:));
    G=Vp;
    Vp = Vp + (epoch-2)/(epoch+1) * (Vp-V);
    V=G;
    fprintf("Reconstructing the tomogram... %d completed out of %d\n",epoch,num_epoch)
end
V = gather(ifft(ifft(V,coordinates.Nx,3),coordinates.Nx,4));